"""
LSTM glucose forecaster for Track B (v4).

Run:  python src/models/train_lstm.py            (full, several minutes)
      python src/models/train_lstm.py --quick    (1 seed, fewer epochs)

v4 changes (each targets a specific weakness):
  1. Gap filter  -- only train/evaluate on windows where the TARGET gap < GAP_MAX_HOURS
     (default 24 h).  Removes the unlearnable multi-day jump noise that was
     dominating the MAE.  Augmented copies of filtered-out windows are also dropped.
  2. Horizon bucket embedding  -- the gap to the predicted reading is bucketed into
     4 classes (0–4 h, 4–12 h, 12–24 h, >24 h) and a 4-dim learned embedding is
     concatenated to the LSTM hidden state before the prediction head.  This lets the
     model learn a different prediction strategy per horizon class instead of squeezing
     everything into a single log-gap scalar.
  3. Patient context vector  -- per-patient summary stats (mean, std, trend slope, last
     reading z-score) computed from all *prior* readings visible in the training fold
     are concatenated to the LSTM hidden state.  This gives the model a patient-level
     prior ("this is a chronically hyperglycaemic patient") without requiring a full
     patient embedding table.

Kept from v3:
  - predicts the CHANGE from the last reading
  - 2-layer LSTM, dropout 0.3, hidden 64
  - L1-style (SmoothL1) loss
  - AdamW + gradient clipping + per-patient early stopping
  - per-patient loss balancing
  - 3-seed ensemble
  - Ridge stacking blend (alpha learned per-fold on training data)
  - synthetic pretraining support (called from train_all.py)
"""

import sys
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MIMIC_TRAJECTORIES, MODELS_SAVED
from src.features.augmentation import jitter
from src.storage.db import init_db, register_model_version, promote_model

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut

# ── experiment flags ─────────────────────────────────────────────────────────
WINDOW          = 8            # 7 readings in, 1 predicted
STRIDE          = 1
JITTER_COPIES   = 4
USE_TIME        = True
BALANCE_PATIENTS = True
HIDDEN          = 64
NUM_LAYERS      = 2
DROPOUT         = 0.3
WEIGHT_DECAY    = 1e-3
LR              = 3e-3
BATCH           = 128
MAX_EPOCHS      = 60
PATIENCE        = 8
SEEDS           = 3
VAL_PATIENTS    = 4

# ── v4 additions ─────────────────────────────────────────────────────────────
GAP_MAX_HOURS   = 24.0          # only predict gaps shorter than this
HORIZON_BINS    = [4.0, 12.0, 24.0]  # bucket edges → 4 buckets
HORIZON_DIM     = 4             # embedding dimension per bucket
CONTEXT_DIM     = 4             # [pat_mean_z, pat_std_z, pat_slope_z, last_z]

LOSS = nn.SmoothL1Loss(reduction="none", beta=0.05)

if "--quick" in sys.argv:
    SEEDS, MAX_EPOCHS = 1, 30


# ── model ─────────────────────────────────────────────────────────────────────

class GlucoseLSTM(nn.Module):
    """
    Inputs
    ------
    x       : (B, WINDOW-1, 2)   scaled glucose + scaled log time-gap per step
    horizon : (B, 1)             log-gap to target (kept for backward compat)
    hbucket : (B,)               long (0-3) horizon bucket index
    ctx     : (B, CONTEXT_DIM)   patient context vector

    Output  : (B,) scaled change from last reading
    """
    def __init__(self, hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT,
                 horizon_dim=HORIZON_DIM, context_dim=CONTEXT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop          = nn.Dropout(dropout)
        self.horizon_embed = nn.Embedding(len(HORIZON_BINS) + 1, horizon_dim)
        # head: LSTM hidden + horizon scalar + horizon embedding + patient context
        head_in = hidden + 1 + horizon_dim + context_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, horizon, hbucket, ctx):
        _, (h_n, _) = self.lstm(x)
        h_last   = self.drop(h_n[-1])                      # (B, hidden)
        h_emb    = self.horizon_embed(hbucket)              # (B, horizon_dim)
        features = torch.cat([h_last, horizon, h_emb, ctx], dim=1)
        return self.head(features).squeeze(-1)


def _horizon_bucket(gap_h: float) -> int:
    for i, edge in enumerate(HORIZON_BINS):
        if gap_h <= edge:
            return i
    return len(HORIZON_BINS)


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(traj_df: pd.DataFrame, gap_max: float = GAP_MAX_HOURS) -> SimpleNamespace:
    """Build sliding-window dataset.

    Each window stores:
      x       (WINDOW-1, 2)   scaled-glucose + log-gap per step
      h       (1,)            log-gap to target
      hbucket int             horizon bucket (0-3)
      ctx     (CONTEXT_DIM,)  patient context  [mean_z, std_z, slope_z, last_z]
      y       float           target glucose
      last    float           last input glucose (for delta recovery)
      groups  patient id
      aug     bool            is augmented copy
    """
    g = traj_df[traj_df["lab_name"] == "glucose"].dropna(subset=["valuenum"]).copy()
    g["charttime"] = pd.to_datetime(g["charttime"])
    rng = np.random.default_rng(0)

    # ── patient-level stats (computed over ALL readings per patient) ──────────
    pat_stats: dict[int, dict] = {}
    for pid, grp in g.groupby("subject_id"):
        v = grp.sort_values("charttime")["valuenum"].to_numpy(float)
        if len(v) < 2:
            slope = 0.0
        else:
            x_idx = np.arange(len(v), dtype=float)
            slope = float(np.polyfit(x_idx, v, 1)[0])
        pat_stats[pid] = {"mean": float(v.mean()), "std": float(v.std()) + 1e-6, "slope": slope}

    # Global stats for z-scoring the context vector
    all_means  = np.array([s["mean"]  for s in pat_stats.values()])
    all_stds   = np.array([s["std"]   for s in pat_stats.values()])
    all_slopes = np.array([s["slope"] for s in pat_stats.values()])
    ctx_mu  = np.array([all_means.mean(),  all_stds.mean(),  all_slopes.mean(), 0.0])
    ctx_sd  = np.array([all_means.std()+1e-6, all_stds.std()+1e-6, all_slopes.std()+1e-6, 1.0])

    X, H, HB, CTX, Y, LAST, groups, aug = [], [], [], [], [], [], [], []

    def add(vals, hrs, pid, is_aug):
        """Commit one window. Skips if target gap > gap_max."""
        gaps = np.clip(np.diff(np.asarray(hrs, float), prepend=hrs[0]), 0, None)
        target_gap = float(gaps[WINDOW - 1])
        if target_gap > gap_max:
            return  # ← gap filter

        vals = np.asarray(vals, float)
        step_gap = np.log1p(gaps[:WINDOW - 1]) / 5 if USE_TIME else np.zeros(WINDOW - 1)

        # patient context (z-scored global stats + last reading z-score within patient)
        ps = pat_stats[pid]
        last_z = (vals[WINDOW - 2] - ps["mean"]) / ps["std"]
        raw_ctx = np.array([ps["mean"], ps["std"], ps["slope"], last_z], dtype=float)
        ctx_vec = (raw_ctx - ctx_mu) / ctx_sd

        X.append(np.stack([vals[:WINDOW - 1], step_gap], axis=1))
        H.append([np.log1p(target_gap) / 5])
        HB.append(_horizon_bucket(target_gap))
        CTX.append(ctx_vec)
        Y.append(vals[-1])
        LAST.append(vals[WINDOW - 2])
        groups.append(pid)
        aug.append(is_aug)

    for pid, grp in g.groupby("subject_id"):
        grp  = grp.sort_values("charttime")
        vals = grp["valuenum"].to_numpy(float)
        hrs  = (grp["charttime"] - grp["charttime"].iloc[0]).dt.total_seconds().to_numpy() / 3600
        for s in range(0, len(vals) - WINDOW + 1, STRIDE):
            v, h = vals[s:s + WINDOW], hrs[s:s + WINDOW]
            add(v, h, pid, False)
            for c in jitter(v, n_copies=JITTER_COPIES,
                            seed=int(rng.integers(1_000_000_000))):
                add(c, h, pid, True)

    D = SimpleNamespace(
        x      = np.array(X,      dtype=float),
        h      = np.array(H,      dtype=float),
        hb     = np.array(HB,     dtype=np.int64),
        ctx    = np.array(CTX,    dtype=float),
        y      = np.array(Y,      dtype=float),
        last   = np.array(LAST,   dtype=float),
        groups = np.array(groups),
        aug    = np.array(aug,    dtype=bool),
        ctx_mu = ctx_mu,
        ctx_sd = ctx_sd,
        pat_stats = pat_stats,
    )
    real = ~D.aug
    D.mu, D.sd = float(D.y[real].mean()), float(D.y[real].std()) if real.sum() > 1 else (150.0, 50.0)
    counts = Counter(D.groups[real])
    D.w = np.array(
        [1 / np.sqrt(counts[p]) if BALANCE_PATIENTS else 1.0 for p in D.groups]
    )
    return D


# ── tensor helpers ────────────────────────────────────────────────────────────

def tensors(D, idx):
    """Convert dataset slice to float32 tensors ready for GlucoseLSTM."""
    xs = D.x[idx].copy()
    xs[:, :, 0] = (xs[:, :, 0] - D.mu) / D.sd
    return (
        torch.tensor(xs,          dtype=torch.float32),
        torch.tensor(D.h[idx],    dtype=torch.float32),
        torch.tensor(D.hb[idx],   dtype=torch.long),
        torch.tensor(D.ctx[idx],  dtype=torch.float32),
        torch.tensor((D.y[idx] - D.last[idx]) / D.sd, dtype=torch.float32),
        torch.tensor(D.w[idx],    dtype=torch.float32),
    )


# ── training ──────────────────────────────────────────────────────────────────

def fit(D, idx_train, idx_val=None, seed=0, fixed_epochs=None):
    torch.manual_seed(seed)
    model = GlucoseLSTM()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    xt, ht, hbt, ctxt, yt, wt = tensors(D, idx_train)
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
            pred = model(xt[b], ht[b], hbt[b], ctxt[b])
            loss = (LOSS(pred, yt[b]) * wt[b]).sum() / wt[b].sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if val is not None:
            model.eval()
            with torch.no_grad():
                vx, vh, vhb, vctx, vy, _ = val
                v = (model(vx, vh, vhb, vctx) - vy).abs().mean().item()
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


# ── inference helpers ─────────────────────────────────────────────────────────

def predict_lstm(models, D, idx):
    """LSTM ensemble prediction in mg/dL."""
    xt, ht, hbt, ctxt, _, _ = tensors(D, idx)
    with torch.no_grad():
        delta = torch.stack([m.eval()(xt, ht, hbt, ctxt) for m in models]).mean(0).numpy()
    return D.last[idx] + delta * D.sd


def flat(D, idx):
    """Ridge feature matrix: window values + simple stats + patient context."""
    w = D.x[idx, :, 0]
    last_diff = w[:, -1] - w[:, -2] if w.shape[1] >= 2 else np.zeros(len(w))
    return np.column_stack([
        w,
        w.mean(1), w.std(1),
        w[:, -1] - w[:, 0],
        last_diff,
        D.ctx[idx],            # v4: patient context added to Ridge too
    ])


def blend_predictions(lstm_preds, ridge_preds, alpha):
    return alpha * lstm_preds + (1.0 - alpha) * ridge_preds


def find_blend_alpha(lstm_preds, ridge_preds, y_true):
    best_alpha, best_mae = 0.5, float("inf")
    for a in np.linspace(0, 1, 21):
        mae = np.abs(blend_predictions(lstm_preds, ridge_preds, a) - y_true).mean()
        if mae < best_mae:
            best_mae, best_alpha = mae, a
    return best_alpha


def summarize(name, errs, pids):
    errs, pids = np.array(errs), np.array(pids)
    macro = np.mean([errs[pids == p].mean() for p in np.unique(pids)])
    print(f"  {name:35s} pooled MAE {errs.mean():6.1f}   per-patient MAE {macro:6.1f}")
    return float(errs.mean())


# ── single-call inference API (used by api_server.py) ─────────────────────────

def predict_from_checkpoint(
    checkpoint_path,
    readings: list,
    gap_hours: list | None = None,
) -> float:
    """Load a saved .pt checkpoint and run a single forecast.

    This is the **only** function api_server.py should call for Track B
    inference. Keeping inference logic here (rather than reimplementing it
    in the server) means architecture changes only need to be made in one
    place.

    Args:
        checkpoint_path: Path to the .pt file produced by train_lstm.py.
        readings:        Recent glucose values (mg/dL). The last (window-1)
                         values are used; extras are ignored.
        gap_hours:       Time gaps between consecutive readings in hours.
                         Pass None or omit to assume uniform spacing.

    Returns:
        Forecasted next glucose reading in mg/dL.

    The function maintains a module-level cache keyed by file mtime so the
    model is only deserialised once per server process (and reloaded
    automatically if the artifact is replaced by a retrain).
    """
    import torch as _torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LSTM artifact not found: {checkpoint_path}")

    # ── lazy load with mtime-based cache ─────────────────────────────────────
    cache = _INFER_CACHE
    mtime = checkpoint_path.stat().st_mtime
    if cache.get("mtime") != mtime:
        ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]

        loaded_models = []
        for state in ckpt["states"]:
            m = GlucoseLSTM(
                hidden      = cfg.get("hidden", HIDDEN),
                num_layers  = cfg.get("num_layers", NUM_LAYERS),
                dropout     = cfg.get("dropout", DROPOUT),
                horizon_dim = cfg.get("horizon_dim", HORIZON_DIM),
                context_dim = cfg.get("context_dim", CONTEXT_DIM),
            )
            m.load_state_dict(state)
            m.eval()
            loaded_models.append(m)

        cache.clear()
        cache["mtime"]           = mtime
        cache["models"]          = loaded_models
        cache["config"]          = cfg
        cache["blend_alpha"]     = float(ckpt.get("blend_alpha", 1.0))
        cache["ridge_coef"]      = ckpt.get("ridge_coef")
        cache["ridge_intercept"] = float(ckpt.get("ridge_intercept") or 0.0)
        cache["ctx_mu"]          = ckpt.get("ctx_mu")
        cache["ctx_sd"]          = ckpt.get("ctx_sd")

    cfg             = cache["config"]
    models          = cache["models"]
    mu: float       = cfg["mu"]
    sd: float       = cfg["sd"]
    window: int     = cfg.get("window", WINDOW)
    use_time: bool  = cfg.get("use_time", True)
    context_dim     = cfg.get("context_dim", 0)
    alpha: float    = cache["blend_alpha"]
    ridge_coef      = cache["ridge_coef"]
    ridge_intercept = cache["ridge_intercept"]
    ctx_mu          = cache["ctx_mu"]
    ctx_sd          = cache["ctx_sd"]

    n_in = window - 1
    vals = np.array(readings[-n_in:], dtype=float)
    last = vals[-1]

    # ── time gaps ─────────────────────────────────────────────────────────────
    if use_time and gap_hours is not None and len(gap_hours) >= n_in:
        raw_gaps = np.array(gap_hours[-n_in:], dtype=float)
    else:
        raw_gaps = np.zeros(n_in, dtype=float)
    step_gap = np.log1p(np.clip(raw_gaps, 0, None)) / 5
    last_gap = float(raw_gaps[-1]) if raw_gaps[-1] > 0 else (
        float(raw_gaps.mean()) if raw_gaps.any() else 1.0
    )

    # ── LSTM input tensor ─────────────────────────────────────────────────────
    scaled_vals = (vals - mu) / sd
    x  = _torch.tensor(
        np.stack([scaled_vals, step_gap], axis=1)[None], dtype=_torch.float32
    )
    h  = _torch.tensor([[np.log1p(last_gap) / 5]], dtype=_torch.float32)

    # ── horizon bucket ────────────────────────────────────────────────────────
    bucket = len(HORIZON_BINS)   # default: longest bin
    for i, edge in enumerate(HORIZON_BINS):
        if last_gap <= edge:
            bucket = i
            break
    hb = _torch.tensor([bucket], dtype=_torch.long)

    # ── patient context vector ────────────────────────────────────────────────
    if context_dim > 0 and ctx_mu is not None and ctx_sd is not None:
        ctx_mu_arr = np.array(ctx_mu, dtype=float)
        ctx_sd_arr = np.array(ctx_sd, dtype=float)
        pat_mean  = float(vals.mean())
        pat_std   = float(vals.std()) + 1e-6
        pat_slope = float(np.polyfit(np.arange(len(vals), dtype=float), vals, 1)[0]) \
                    if len(vals) >= 2 else 0.0
        last_z    = (vals[-1] - pat_mean) / pat_std
        ctx_vec   = (np.array([pat_mean, pat_std, pat_slope, last_z]) - ctx_mu_arr) / ctx_sd_arr
        ctx       = _torch.tensor(ctx_vec[None], dtype=_torch.float32)
    else:
        ctx_vec = None
        ctx     = _torch.zeros(1, max(context_dim, 4), dtype=_torch.float32)

    # ── forward pass ──────────────────────────────────────────────────────────
    with _torch.no_grad():
        delta_scaled = _torch.stack(
            [m(x, h, hb, ctx) for m in models]
        ).mean(0).item()

    lstm_forecast = float(last + delta_scaled * sd)

    # ── Ridge blend ───────────────────────────────────────────────────────────
    if ridge_coef is not None and alpha < 1.0:
        last_diff = float(vals[-1] - vals[-2]) if len(vals) >= 2 else 0.0
        base_feats = np.concatenate([
            vals,
            [vals.mean(), vals.std(), vals[-1] - vals[0], last_diff],
        ])
        ridge_feats = (
            np.concatenate([base_feats, ctx_vec])
            if ctx_vec is not None else base_feats
        )
        ridge_forecast = float(np.dot(ridge_coef, ridge_feats) + ridge_intercept)
        return alpha * lstm_forecast + (1.0 - alpha) * ridge_forecast

    return lstm_forecast


# Module-level inference cache — shared across all calls within one process.
# Populated lazily by predict_from_checkpoint(); cleared on file change.
_INFER_CACHE: dict = {}


# ── standalone entry point ────────────────────────────────────────────────────

def main():
    D    = load_data(pd.read_csv(MIMIC_TRAJECTORIES))
    real = ~D.aug
    n_real_windows = int(real.sum())
    print(f"Real patients: {len(set(D.groups))} | real windows (gap<{GAP_MAX_HOURS}h): {n_real_windows} | "
          f"total with augmentation: {len(D.y)}")
    print(f"Flags: window={WINDOW} gap_max={GAP_MAX_HOURS}h hidden={HIDDEN} "
          f"layers={NUM_LAYERS} dropout={DROPOUT} seeds={SEEDS}")

    rng       = np.random.default_rng(42)
    errs      = {"Naive": [], "Ridge": [], "LSTM v4": [], "LSTM v4 + Ridge": []}
    pids      = []
    best_epochs = []
    fold_alphas = []

    for fold, (tr, te) in enumerate(
            LeaveOneGroupOut().split(D.x, D.y, D.groups), 1):
        te = te[real[te]]
        if len(te) == 0:
            continue

        train_pats = np.unique(D.groups[tr])
        val_pats   = rng.choice(
            train_pats,
            size=min(VAL_PATIENTS, len(train_pats) - 1),
            replace=False,
        )
        in_val  = np.isin(D.groups[tr], val_pats)
        tr_fit  = tr[~in_val]
        tr_val  = tr[in_val & real[tr]]

        ridge_fold = Ridge(alpha=1.0).fit(flat(D, tr), D.y[tr])

        models_fold = []
        for s in range(SEEDS):
            m, ep = fit(D, tr_fit, tr_val, seed=s)
            models_fold.append(m)
            best_epochs.append(ep)

        y_te      = D.y[te]
        lstm_te   = predict_lstm(models_fold, D, te)
        ridge_te  = ridge_fold.predict(flat(D, te))

        # blend alpha found on training data only (no leakage)
        real_tr  = tr_fit[real[tr_fit]]
        alpha    = find_blend_alpha(
            predict_lstm(models_fold, D, real_tr),
            ridge_fold.predict(flat(D, real_tr)),
            D.y[real_tr],
        )
        fold_alphas.append(alpha)

        errs["Naive"].extend(np.abs(D.last[te] - y_te))
        errs["Ridge"].extend(np.abs(ridge_te - y_te))
        errs["LSTM v4"].extend(np.abs(lstm_te - y_te))
        errs["LSTM v4 + Ridge"].extend(
            np.abs(blend_predictions(lstm_te, ridge_te, alpha) - y_te)
        )
        pids.extend(D.groups[te])

        if fold % 5 == 0:
            print(f"  ... {fold} folds done")

    mean_alpha = float(np.mean(fold_alphas))
    print(f"\n  Average blend alpha (LSTM weight): {mean_alpha:.2f}")
    print(f"\nLeave-one-patient-out results  (gap < {GAP_MAX_HOURS}h only, mg/dL ↓):")
    res = {name: summarize(name, e, pids) for name, e in errs.items()}

    # ── final model ───────────────────────────────────────────────────────────
    final_epochs = max(1, int(np.median(best_epochs)))
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
            "states":           [m.state_dict() for m in finals],
            "ridge_coef":       final_ridge.coef_.tolist(),
            "ridge_intercept":  float(final_ridge.intercept_),
            "blend_alpha":      mean_alpha,
            "ctx_mu":           D.ctx_mu.tolist(),
            "ctx_sd":           D.ctx_sd.tolist(),
            "config": {
                "mu":           D.mu,
                "sd":           D.sd,
                "hidden":       HIDDEN,
                "num_layers":   NUM_LAYERS,
                "dropout":      DROPOUT,
                "use_time":     USE_TIME,
                "window":       WINDOW,
                "gap_max":      GAP_MAX_HOURS,
                "horizon_dim":  HORIZON_DIM,
                "context_dim":  CONTEXT_DIM,
                "ctx_mu":       D.ctx_mu.tolist(),
                "ctx_sd":       D.ctx_sd.tolist(),
            },
        },
        path,
    )
    print(f"Saved {SEEDS}-model LSTM v4 + Ridge blend (alpha={mean_alpha:.2f}) -> {path}")

    init_db()
    best_name = "LSTM v4 + Ridge" if res["LSTM v4 + Ridge"] <= res["LSTM v4"] else "LSTM v4"
    best_mae  = res[best_name]
    version_id = register_model_version(
        model_name="mimic_glucose_forecaster_lstm",
        n_original_rows=n_real_windows, n_app_rows=0,
        val_auc=0.0, val_f1=0.0, val_mae=best_mae,
        notes=(
            f"LSTM v4: gap_filter={GAP_MAX_HOURS}h + horizon_embed + patient_ctx, "
            f"window={WINDOW}, hidden={HIDDEN}, layers={NUM_LAYERS}, dropout={DROPOUT}, "
            f"seeds={SEEDS}, alpha={mean_alpha:.2f}, "
            f"MAE {best_mae:.1f} mg/dL (pooled LOPO-CV, short-gap only)"
        ),
    )
    promote_model(version_id, "mimic_glucose_forecaster_lstm")
    print(f"Registered as mimic_glucose_forecaster_lstm, version {version_id}")


if __name__ == "__main__":
    main()
