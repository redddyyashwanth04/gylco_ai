"""
LSTM glucose forecaster for Track B (v3).

Run:  python src/models/train_lstm.py            (full, several minutes)
      python src/models/train_lstm.py --quick    (1 seed, fewer epochs)

Changes vs v2 (each targets a specific weakness):
  - Window 5 -> 8  (7 readings in, 1 predicted – more history)
  - 2-layer LSTM   (richer temporal representation)
  - Dropout 0.2 -> 0.3  (stronger regularisation on 30 patients)
  - LSTM + Ridge stacking ensemble  (blends both at inference)
    The blend weight alpha is learned per-fold and averaged for the final model.

Already in v2 (kept):
  - predicts the CHANGE from the last reading  (starts at the naive baseline)
  - L1-style loss
  - time gaps as inputs
  - AdamW + gradient clipping + early stopping on held-out patients
  - per-patient loss balancing
  - ensemble of seeds
"""

import sys
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MIMIC_TRAJECTORIES, MODELS_SAVED
from src.features.augmentation import jitter, time_warp
from src.storage.db import init_db, register_model_version, promote_model

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut

# ---- experiment flags -------------------------------------------------------
WINDOW        = 8          # 7 readings in, 1 predicted (was 5)
STRIDE        = 1
JITTER_COPIES = 4
USE_TIME_WARP = False
TIME_WARP_COPIES = 2
USE_TIME      = True
BALANCE_PATIENTS = True
HIDDEN        = 64         # was 32 in v2
NUM_LAYERS    = 2          # was 1 in v2
DROPOUT       = 0.3        # was 0.2 in v2
WEIGHT_DECAY  = 1e-3
LR            = 3e-3
BATCH         = 128
MAX_EPOCHS    = 60
PATIENCE      = 8
SEEDS         = 3
VAL_PATIENTS  = 4
LOSS          = nn.SmoothL1Loss(reduction="none", beta=0.05)

if "--quick" in sys.argv:
    SEEDS, MAX_EPOCHS = 1, 30


class GlucoseLSTM(nn.Module):
    """
    Input  (batch, WINDOW-1, 2): [scaled glucose, scaled log time-gap]
    Output: scaled change from last reading (scalar per sample)

    v3 changes: 2-layer LSTM, dropout 0.3, accepts WINDOW-1 steps (configurable).
    """
    def __init__(self, hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # inter-layer dropout
        )
        self.drop = nn.Dropout(dropout)   # output dropout before head
        self.head = nn.Linear(hidden + 1, 1)  # +1 = log time-gap to target

    def forward(self, x, horizon):
        _, (h_n, _) = self.lstm(x)          # h_n: (num_layers, B, hidden)
        h = torch.cat([self.drop(h_n[-1]), horizon], dim=1)
        return self.head(h).squeeze(-1)


def load_data(traj_df):
    g = traj_df[traj_df["lab_name"] == "glucose"].dropna(subset=["valuenum"]).copy()
    g["charttime"] = pd.to_datetime(g["charttime"])
    rng = np.random.default_rng(0)
    X, H, Y, LAST, groups, aug = [], [], [], [], [], []

    def add(vals, hrs, pid, is_aug):
        vals = np.asarray(vals, float)
        gaps = np.clip(np.diff(np.asarray(hrs, float), prepend=hrs[0]), 0, None)
        if not USE_TIME:
            gaps = np.zeros_like(gaps)
        step_gap = np.log1p(gaps[:WINDOW - 1]) / 5
        X.append(np.stack([vals[:WINDOW - 1], step_gap], axis=1))
        H.append([np.log1p(gaps[WINDOW - 1]) / 5])
        Y.append(vals[-1])
        LAST.append(vals[WINDOW - 2])
        groups.append(pid)
        aug.append(is_aug)

    for pid, grp in g.groupby("subject_id"):
        grp = grp.sort_values("charttime")
        vals = grp["valuenum"].to_numpy(float)
        hrs  = (grp["charttime"] - grp["charttime"].iloc[0]).dt.total_seconds().to_numpy() / 3600
        for s in range(0, len(vals) - WINDOW + 1, STRIDE):
            v, h = vals[s:s + WINDOW], hrs[s:s + WINDOW]
            add(v, h, pid, False)
            for c in jitter(v, n_copies=JITTER_COPIES,
                            seed=int(rng.integers(1_000_000_000))):
                add(c, h, pid, True)
            if USE_TIME_WARP:
                for _ in range(TIME_WARP_COPIES):
                    add(v, time_warp(h - h[0],
                        seed=int(rng.integers(1_000_000_000))), pid, True)

    D = SimpleNamespace(
        x=np.array(X), h=np.array(H), y=np.array(Y), last=np.array(LAST),
        groups=np.array(groups), aug=np.array(aug),
    )
    real = ~D.aug
    D.mu, D.sd = float(D.y[real].mean()), float(D.y[real].std())
    counts = Counter(D.groups[real])
    D.w = np.array(
        [1 / np.sqrt(counts[p]) if BALANCE_PATIENTS else 1.0 for p in D.groups]
    )
    return D


def tensors(D, idx):
    xs = D.x[idx].copy()
    xs[:, :, 0] = (xs[:, :, 0] - D.mu) / D.sd
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(D.h[idx], dtype=torch.float32),
        torch.tensor((D.y[idx] - D.last[idx]) / D.sd, dtype=torch.float32),
        torch.tensor(D.w[idx], dtype=torch.float32),
    )


def fit(D, idx_train, idx_val=None, seed=0, fixed_epochs=None):
    torch.manual_seed(seed)
    model = GlucoseLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xt, ht, yt, wt = tensors(D, idx_train)
    val = tensors(D, idx_val) if idx_val is not None and len(idx_val) else None
    n   = len(yt)
    best_v, best_state, best_ep, bad = float("inf"), None, 0, 0
    epochs = fixed_epochs or MAX_EPOCHS
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = (LOSS(model(xt[b], ht[b]), yt[b]) * wt[b]).sum() / wt[b].sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val is not None:
            model.eval()
            with torch.no_grad():
                v = (model(val[0], val[1]) - val[2]).abs().mean().item()
            if v < best_v - 1e-4:
                best_v, best_ep, bad = v, ep, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_ep or epochs)


def predict_lstm(models, D, idx):
    """LSTM ensemble raw prediction (mg/dL)."""
    xs, hs, _, _ = tensors(D, idx)
    with torch.no_grad():
        delta = torch.stack([m.eval()(xs, hs) for m in models]).mean(0).numpy()
    return D.last[idx] + delta * D.sd


def flat(D, idx):
    """Feature matrix for Ridge: raw window values + simple stats."""
    w = D.x[idx, :, 0]
    # trend features: last-first, mean, std, last value, last two diff
    last_diff = w[:, -1] - w[:, -2] if w.shape[1] >= 2 else np.zeros(len(w))
    return np.column_stack([
        w,
        w.mean(1), w.std(1),
        w[:, -1] - w[:, 0],   # overall trend
        last_diff,             # local momentum
    ])


def blend_predictions(lstm_preds, ridge_preds, alpha):
    """alpha * LSTM + (1-alpha) * Ridge."""
    return alpha * lstm_preds + (1.0 - alpha) * ridge_preds


def find_blend_alpha(lstm_preds, ridge_preds, y_true):
    """Grid-search alpha in [0,1] that minimises MAE on the given fold."""
    best_alpha, best_mae = 0.5, float("inf")
    for a in np.linspace(0, 1, 21):
        mae = np.abs(blend_predictions(lstm_preds, ridge_preds, a) - y_true).mean()
        if mae < best_mae:
            best_mae, best_alpha = mae, a
    return best_alpha


def summarize(name, errs, pids):
    errs, pids = np.array(errs), np.array(pids)
    macro = np.mean([errs[pids == p].mean() for p in np.unique(pids)])
    print(f"  {name:30s} pooled MAE {errs.mean():6.1f}   per-patient MAE {macro:6.1f}")
    return float(errs.mean())


def main():
    D = load_data(pd.read_csv(MIMIC_TRAJECTORIES))
    real = ~D.aug
    print(f"Real patients: {len(set(D.groups))} | real windows: {int(real.sum())} | "
          f"total with augmentation: {len(D.y)}")
    print(f"Flags: window={WINDOW} time={USE_TIME} balance={BALANCE_PATIENTS} "
          f"hidden={HIDDEN} layers={NUM_LAYERS} dropout={DROPOUT} seeds={SEEDS}")

    rng = np.random.default_rng(42)
    errs = {
        "Naive (last value)":  [],
        "Ridge":               [],
        "LSTM v3":             [],
        "LSTM v3 + Ridge":     [],
    }
    pids          = []
    best_epochs   = []
    fold_alphas   = []

    for fold, (tr, te) in enumerate(
            LeaveOneGroupOut().split(D.x, D.y, D.groups), 1):
        te = te[real[te]]
        if len(te) == 0:
            continue
        train_patients = np.unique(D.groups[tr])
        val_patients   = rng.choice(
            train_patients,
            size=min(VAL_PATIENTS, len(train_patients) - 1),
            replace=False,
        )
        in_val  = np.isin(D.groups[tr], val_patients)
        tr_fit  = tr[~in_val]
        tr_val  = tr[in_val & real[tr]]

        # Ridge
        ridge_fold = Ridge(alpha=1.0).fit(flat(D, tr), D.y[tr])

        # LSTM
        models_fold = []
        for s in range(SEEDS):
            m, ep = fit(D, tr_fit, tr_val, seed=s)
            models_fold.append(m)
            best_epochs.append(ep)

        y_te         = D.y[te]
        lstm_te      = predict_lstm(models_fold, D, te)
        ridge_te     = ridge_fold.predict(flat(D, te))

        # find best alpha on *training* fold (not test) to avoid leakage
        lstm_tr  = predict_lstm(models_fold, D, tr_fit[real[tr_fit]])
        ridge_tr = ridge_fold.predict(flat(D, tr_fit[real[tr_fit]]))
        y_tr     = D.y[tr_fit[real[tr_fit]]]
        alpha    = find_blend_alpha(lstm_tr, ridge_tr, y_tr)
        fold_alphas.append(alpha)

        errs["Naive (last value)"].extend(np.abs(D.last[te] - y_te))
        errs["Ridge"].extend(np.abs(ridge_te - y_te))
        errs["LSTM v3"].extend(np.abs(lstm_te - y_te))
        errs["LSTM v3 + Ridge"].extend(
            np.abs(blend_predictions(lstm_te, ridge_te, alpha) - y_te)
        )
        pids.extend(D.groups[te])

        if fold % 5 == 0:
            print(f"  ... {fold} folds done")

    mean_alpha = float(np.mean(fold_alphas))
    print(f"\n  Average blend alpha (LSTM weight): {mean_alpha:.2f}")
    print("\nLeave-one-patient-out results (mg/dL, lower is better):")
    res = {name: summarize(name, e, pids) for name, e in errs.items()}
    print("Pooled = comparable to earlier numbers. Per-patient = every patient counts equally.")

    # --- final model: train on all data --------------------------------------
    final_epochs = int(np.median(best_epochs))
    all_idx      = np.arange(len(D.y))

    print(f"\nTraining final ensemble ({SEEDS} seeds, {final_epochs} epochs) ...")
    finals = [
        fit(D, all_idx, None, seed=s, fixed_epochs=final_epochs)[0]
        for s in range(SEEDS)
    ]
    final_ridge = Ridge(alpha=1.0).fit(flat(D, all_idx), D.y[all_idx])

    MODELS_SAVED.mkdir(parents=True, exist_ok=True)
    path = MODELS_SAVED / "mimic_lstm_forecaster.pt"
    torch.save(
        {
            "states": [m.state_dict() for m in finals],
            "ridge_coef":  final_ridge.coef_.tolist(),
            "ridge_intercept": float(final_ridge.intercept_),
            "blend_alpha": mean_alpha,
            "config": {
                "mu":      D.mu,
                "sd":      D.sd,
                "hidden":  HIDDEN,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "use_time": USE_TIME,
                "window":  WINDOW,
            },
        },
        path,
    )
    print(f"Saved {SEEDS}-model LSTM + Ridge blend (alpha={mean_alpha:.2f}) -> {path}")

    init_db()
    best_name = "LSTM v3 + Ridge" if res["LSTM v3 + Ridge"] <= res["LSTM v3"] else "LSTM v3"
    best_mae  = res[best_name]
    version_id = register_model_version(
        model_name="mimic_glucose_forecaster_lstm",
        n_original_rows=int(real.sum()), n_app_rows=0,
        val_auc=0.0, val_f1=0.0,
        notes=(
            f"LSTM v3 + Ridge blend, window={WINDOW}, hidden={HIDDEN}, "
            f"layers={NUM_LAYERS}, dropout={DROPOUT}, seeds={SEEDS}, "
            f"alpha={mean_alpha:.2f}, "
            f"mean abs error {best_mae:.1f} mg/dL (pooled LOPO-CV)"
        ),
    )
    promote_model(version_id, "mimic_glucose_forecaster_lstm")
    print(f"Registered as mimic_glucose_forecaster_lstm, version {version_id}")


if __name__ == "__main__":
    main()
