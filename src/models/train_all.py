"""
Unified training script -- Train Track A and Track B in one run.

Run:
    python src/models/train_all.py           # full (several minutes for LSTM)
    python src/models/train_all.py --quick   # fewer LSTM epochs (useful for CI)

What it does
------------
Track A (NHANES -- complication risk)
  Trains three classifiers on nhanes_model_ready.csv:
    1. Logistic Regression  (traditional baseline)
    2. Random Forest        (nonlinear)
    3. Gradient Boosting    (nonlinear)
  Best nonlinear model by mean AUC on the held-out 20% is promoted as the
  active model served by /api/predict.

Track B (MIMIC-IV -- glucose forecast)
  Trains three forecasters via leave-one-patient-out CV on the same windows:
    1. Naive (repeat last value)
    2. Ridge
    3. LSTM v2 residual ensemble
  Best model by pooled LOPO MAE (lower is better) is promoted as the active
  model served by /api/forecast.
  - If LSTM wins: saved to  models_saved/mimic_lstm_forecaster.pt
                  registry key: mimic_glucose_forecaster_lstm
  - If Ridge wins: saved to models_saved/mimic_glucose_forecaster.pkl
                   registry key: mimic_glucose_forecaster
  The api_server always tries the LSTM .pt first, Ridge .pkl as fallback,
  so whichever artifact is freshest/best wins automatically.
"""

import pickle
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_MODEL_READY, MIMIC_TRAJECTORIES, MODELS_SAVED
from src.features.augmentation import jitter
from src.storage.db import init_db, promote_model, register_model_version

# -- shared constants ---------------------------------------------------------
QUICK = "--quick" in sys.argv

# =============================================================================
# TRACK A -- NHANES complication risk
# =============================================================================

TARGET_COLUMNS = ["hypertension", "nephropathy", "cardiovascular"]
LEAKAGE_COLUMNS = [
    "BPQ020", "KIQ022", "URXUMA", "MCQ160B", "MCQ160C",
    "MCQ160E", "MCQ160F", "OBESITY_FLAG",
]
DROP_COLUMNS = ["SEQN", "diabetes", "obesity"] + TARGET_COLUMNS + LEAKAGE_COLUMNS


def _load_nhanes():
    path = DATA_MODEL_READY / "nhanes_model_ready.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: preprocess_nhanes.py -> feature_engineering.py -> build_targets.py"
        )
    return pd.read_csv(path)


def _split_nhanes(df):
    from sklearn.model_selection import train_test_split
    feat_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df[TARGET_COLUMNS]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_te, y_tr, y_te, feat_cols


def _eval_a(model, X_te, y_te):
    from sklearn.metrics import f1_score, roc_auc_score
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)
    aucs, f1s = [], []
    for i, col in enumerate(TARGET_COLUMNS):
        try:
            auc = roc_auc_score(y_te[col], y_proba[i][:, 1])
            aucs.append(auc)
        except ValueError:
            auc = 0.0
        f1 = f1_score(y_te[col], y_pred[:, i], zero_division=0)
        f1s.append(f1)
        print(f"    {col:15s}  AUC={round(auc, 3):<6}  F1={round(f1, 3)}")
    return float(np.mean(aucs)) if aucs else 0.0, float(np.mean(f1s))


def _save_pkl(obj, name: str) -> Path:
    MODELS_SAVED.mkdir(parents=True, exist_ok=True)
    p = MODELS_SAVED / f"{name}.pkl"
    with open(p, "wb") as fh:
        pickle.dump(obj, fh)
    return p


def train_track_a():
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("TRACK A -- NHANES complication risk")
    print("=" * 60)

    df = _load_nhanes()
    X_tr, X_te, y_tr, y_te, feat_cols = _split_nhanes(df)
    print(f"Train: {len(X_tr)} rows | Test: {len(X_te)} rows | Features: {len(feat_cols)}")

    # 1. Logistic Regression
    print("\n  Logistic Regression (traditional baseline)")
    lr = MultiOutputClassifier(
        make_pipeline(StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=2000))
    ).fit(X_tr, y_tr)
    lr_auc, lr_f1 = _eval_a(lr, X_te, y_te)
    _save_pkl(lr, "nhanes_traditional_baseline")
    print(f"  -> mean AUC={round(lr_auc, 3)}  F1={round(lr_f1, 3)}")

    # 2. Random Forest
    print("\n  Random Forest")
    rf = MultiOutputClassifier(
        RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=42)
    ).fit(X_tr, y_tr)
    rf_auc, rf_f1 = _eval_a(rf, X_te, y_te)
    _save_pkl(rf, "nhanes_random_forest")
    print(f"  -> mean AUC={round(rf_auc, 3)}  F1={round(rf_f1, 3)}")

    # 3. Gradient Boosting
    print("\n  Gradient Boosting")
    gb = MultiOutputClassifier(
        HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    ).fit(X_tr, y_tr)
    gb_auc, gb_f1 = _eval_a(gb, X_te, y_te)
    _save_pkl(gb, "nhanes_gradient_boosting")
    print(f"  -> mean AUC={round(gb_auc, 3)}  F1={round(gb_f1, 3)}")

    # Comparison table
    print("\n  +--------------------------------------------------+")
    print(f"  | Logistic Regression  AUC={round(lr_auc,3):<6}  F1={round(lr_f1,3):<6}  |")
    print(f"  | Random Forest        AUC={round(rf_auc,3):<6}  F1={round(rf_f1,3):<6}  |")
    print(f"  | Gradient Boosting    AUC={round(gb_auc,3):<6}  F1={round(gb_f1,3):<6}  |")
    print("  +--------------------------------------------------+")

    # Promote best nonlinear model by AUC
    best_name, best_auc, best_f1 = max(
        [("nhanes_random_forest",     rf_auc, rf_f1),
         ("nhanes_gradient_boosting", gb_auc, gb_f1)],
        key=lambda t: t[1],
    )
    version_id = register_model_version(
        model_name="nhanes_active",
        n_original_rows=len(X_tr), n_app_rows=0,
        val_auc=best_auc, val_f1=best_f1,
        notes=f"best nonlinear model: {best_name}",
    )
    promote_model(version_id, "nhanes_active")
    (MODELS_SAVED / "nhanes_active_name.txt").write_text(best_name)
    with open(MODELS_SAVED / "nhanes_feature_columns.pkl", "wb") as fh:
        pickle.dump(feat_cols, fh)

    print(f"\n  [OK] Active Track A model -> {best_name}  (registry v{version_id})")
    return best_name, best_auc, best_f1


# =============================================================================
# TRACK B -- MIMIC-IV glucose forecast
# =============================================================================

# LSTM hyper-parameters -- kept in sync with train_lstm.py
WINDOW        = 8            # v3: 7 readings in, 1 predicted (was 5)
STRIDE        = 1
JITTER_COPIES = 4
USE_TIME      = True
BALANCE       = True
HIDDEN        = 64
NUM_LAYERS    = 2            # v3: 2-layer LSTM
DROPOUT       = 0.3          # v3: stronger regularisation
WEIGHT_DECAY  = 1e-3
LR_PRETRAIN   = 5e-3
LR_FINETUNE   = 1e-3
BATCH         = 256
MAX_EPOCHS_PRETRAIN = 20 if QUICK else 40
MAX_EPOCHS_FINETUNE = 20 if QUICK else 40
PATIENCE      = 8
SEEDS         = 1 if QUICK else 3
VAL_PATIENTS  = 4
SYN_PATH      = Path(__file__).resolve().parent.parent.parent / "data" / "augmented" / "synthetic_mimic_trajectories.csv"


def _build_windows(traj_df: pd.DataFrame, rng: np.random.Generator) -> SimpleNamespace:
    """Build sliding-window dataset from a trajectories dataframe.
    Works for both real and synthetic data (same schema)."""
    g = traj_df[traj_df["lab_name"] == "glucose"].dropna(subset=["valuenum"]).copy()
    g["charttime"] = pd.to_datetime(g["charttime"])
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

    D = SimpleNamespace(
        x=np.array(X), h=np.array(H), y=np.array(Y), last=np.array(LAST),
        groups=np.array(groups), aug=np.array(aug),
    )
    real = ~D.aug
    D.mu  = float(D.y[real].mean())
    D.sd  = float(D.y[real].std())
    from collections import Counter
    counts = Counter(D.groups[real])
    D.w = np.array(
        [1 / np.sqrt(counts[p]) if BALANCE else 1.0 for p in D.groups]
    )
    return D


def _load_mimic() -> SimpleNamespace:
    """Build sliding-window dataset from the real MIMIC trajectories only."""
    return _build_windows(pd.read_csv(MIMIC_TRAJECTORIES), np.random.default_rng(0))


# Import GlucoseLSTM from the canonical source so both scripts share one definition
try:
    import torch
    import torch.nn as nn
    from src.models.train_lstm import (
        GlucoseLSTM,
        flat as _flat_ridge_fn,
        blend_predictions,
        find_blend_alpha,
    )
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    GlucoseLSTM = None
    _flat_ridge_fn = None
    blend_predictions = None
    find_blend_alpha = None


def _tensors(D, idx):
    import torch
    xs = D.x[idx].copy()
    xs[:, :, 0] = (xs[:, :, 0] - D.mu) / D.sd
    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(D.h[idx], dtype=torch.float32),
        torch.tensor((D.y[idx] - D.last[idx]) / D.sd, dtype=torch.float32),
        torch.tensor(D.w[idx], dtype=torch.float32),
    )


def _fit_lstm(D, idx_train, idx_val=None, seed=0, fixed_epochs=None,
              lr=None, pretrained_state=None):
    """Train one LSTM seed. Optionally warm-start from pretrained_state."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    LOSS = nn.SmoothL1Loss(reduction="none", beta=0.05)
    model = GlucoseLSTM(hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT)
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state, strict=False)
    actual_lr = lr if lr is not None else LR_FINETUNE
    opt = torch.optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=WEIGHT_DECAY)
    xt, ht, yt, wt = _tensors(D, idx_train)
    val = _tensors(D, idx_val) if idx_val is not None and len(idx_val) else None
    n   = len(yt)
    best_v, best_state, best_ep, bad = float("inf"), None, 0, 0
    epochs = fixed_epochs or MAX_EPOCHS_FINETUNE
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


def _pretrain_on_synthetic(seed: int = 0):
    """Stage 1: pretrain LSTM on 1000 synthetic patients. Returns state_dict."""
    import torch
    import torch.nn as nn
    if not SYN_PATH.exists():
        print("  [Pretrain] synthetic_mimic_trajectories.csv not found -- skipping pretrain.")
        print("  Run: python src/data_prep/generate_synthetic_mimic.py")
        return None
    print(f"  [Pretrain] Loading synthetic data ...")
    syn_df = pd.read_csv(SYN_PATH)
    if "is_synthetic" in syn_df.columns:
        syn_df = syn_df.drop(columns=["is_synthetic"])
    D_syn = _build_windows(syn_df, np.random.default_rng(seed))
    all_pids = np.unique(D_syn.groups)
    rng = np.random.default_rng(seed)
    val_pids = rng.choice(all_pids, size=max(1, len(all_pids) // 20), replace=False)
    in_val   = np.isin(D_syn.groups, val_pids)
    tr_idx   = np.where(~in_val)[0]
    va_idx   = np.where(in_val)[0]
    print(f"  [Pretrain] {len(D_syn.y)} windows | {len(all_pids)} syn patients "
          f"| val={len(va_idx)} windows")
    torch.manual_seed(seed)
    LOSS = nn.SmoothL1Loss(reduction="none", beta=0.05)
    model = GlucoseLSTM(hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
    xt, ht, yt, wt = _tensors(D_syn, tr_idx)
    vxt, vht, vyt, _ = _tensors(D_syn, va_idx)
    n = len(yt)
    best_v, best_state, bad = float("inf"), None, 0
    for ep in range(1, MAX_EPOCHS_PRETRAIN + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = (LOSS(model(xt[b], ht[b]), yt[b]) * wt[b]).sum() / wt[b].sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = (model(vxt, vht) - vyt).abs().mean().item()
        if v < best_v - 1e-4:
            best_v, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    print(f"  [Pretrain] Done. Best val MAE ~{best_v * D_syn.sd:.1f} mg/dL (synthetic scale)")
    return best_state


def _predict_lstm(models, D, idx):
    import torch
    xs, hs, _, _ = _tensors(D, idx)
    with torch.no_grad():
        delta = torch.stack([m.eval()(xs, hs) for m in models]).mean(0).numpy()
    return D.last[idx] + delta * D.sd


def _flat_ridge(D, idx):
    # Delegates to train_lstm.flat() so both scripts share one definition.
    # Falls back to the simple version if torch/train_lstm not yet importable.
    if _flat_ridge_fn is not None:
        return _flat_ridge_fn(D, idx)
    w = D.x[idx, :, 0]
    last_diff = w[:, -1] - w[:, -2] if w.shape[1] >= 2 else np.zeros(len(w))
    return np.column_stack([w, w.mean(1), w.std(1), w[:, -1] - w[:, 0], last_diff])


def _summarize_b(name, errs, pids):
    errs, pids = np.array(errs), np.array(pids)
    macro  = np.mean([errs[pids == p].mean() for p in np.unique(pids)])
    pooled = float(errs.mean())
    print(f"  {name:28s}  pooled MAE {pooled:6.1f}   per-patient MAE {macro:6.1f}")
    return pooled


def train_track_b():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneGroupOut

    print("\n" + "=" * 60)
    print("TRACK B -- MIMIC-IV glucose forecast (pretrain -> fine-tune)")
    print("=" * 60)

    if not MIMIC_TRAJECTORIES.exists():
        print(f"  [X] {MIMIC_TRAJECTORIES} not found -- skipping Track B.")
        return None, None

    if not TORCH_OK:
        print("  [X] torch not installed -- LSTM skipped. Run: pip install torch")
        print("  Falling back to Ridge only.")

    D = _load_mimic()
    real = ~D.aug
    print(f"Real patients: {len(set(D.groups))} | real windows: {int(real.sum())} | "
          f"total with augmentation: {len(D.y)}")

    # Stage 1: pretrain on synthetic data (once, shared across seeds/folds)
    pretrained_state = None
    if TORCH_OK:
        print(f"\nStage 1: Pretrain on synthetic data (hidden={HIDDEN})")
        pretrained_state = _pretrain_on_synthetic(seed=0)
        print(f"\nStage 2: Fine-tune LOPO-CV on {len(set(D.groups))} real patients "
              f"(seeds={SEEDS}, max_epochs={MAX_EPOCHS_FINETUNE})")

    rng = np.random.default_rng(42)
    model_names = ["Naive (last value)", "Ridge"] + (["LSTM v3 + pretrain", "LSTM v3 + Ridge blend"] if TORCH_OK else [])
    errs  = {n: [] for n in model_names}
    pids  = []
    best_epochs_list = []
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

        # Ridge
        ridge_fold = Ridge(alpha=1.0).fit(_flat_ridge(D, tr), D.y[tr])

        # LSTM fine-tune from pretrained weights
        if TORCH_OK:
            models_fold = []
            for s in range(SEEDS):
                m, ep = _fit_lstm(D, tr_fit, tr_val, seed=s,
                                  lr=LR_FINETUNE,
                                  pretrained_state=pretrained_state)
                models_fold.append(m)
                best_epochs_list.append(ep)

        y_te      = D.y[te]
        ridge_te  = ridge_fold.predict(_flat_ridge(D, te))
        errs["Naive (last value)"].extend(np.abs(D.last[te] - y_te))
        errs["Ridge"].extend(np.abs(ridge_te - y_te))
        if TORCH_OK:
            lstm_te = _predict_lstm(models_fold, D, te)
            errs["LSTM v3 + pretrain"].extend(np.abs(lstm_te - y_te))

            # find blend alpha on training data (no leakage)
            real_tr_fit = tr_fit[real[tr_fit]]
            lstm_tr  = _predict_lstm(models_fold, D, real_tr_fit)
            ridge_tr = ridge_fold.predict(_flat_ridge(D, real_tr_fit))
            y_tr     = D.y[real_tr_fit]
            alpha    = find_blend_alpha(lstm_tr, ridge_tr, y_tr)
            fold_alphas.append(alpha)
            errs["LSTM v3 + Ridge blend"].extend(
                np.abs(blend_predictions(lstm_te, ridge_te, alpha) - y_te)
            )

        pids.extend(D.groups[te])

        if fold % 5 == 0:
            print(f"  ... {fold} folds done")

    print("\n  Leave-one-patient-out results (mg/dL, lower is better):")
    mae_results = {name: _summarize_b(name, e, pids) for name, e in errs.items()}
    print("  Pooled = comparable across runs. Per-patient = every patient counts equally.")

    # Determine winner
    scored = {k: v for k, v in mae_results.items() if k != "Naive (last value)"}
    winner = min(scored, key=scored.get)
    print(f"\n  Winner: {winner}  (pooled MAE {scored[winner]:.1f} mg/dL)")
    print(f"  vs naive baseline: {mae_results['Naive (last value)']:.1f} mg/dL")
    improvement = mae_results["Naive (last value)"] - scored[winner]
    print(f"  Improvement over naive: {improvement:.1f} mg/dL "
          f"({improvement / mae_results['Naive (last value)'] * 100:.0f}%)")

    MODELS_SAVED.mkdir(parents=True, exist_ok=True)

    if winner.startswith("LSTM"):
        import torch
        mean_alpha   = float(np.mean(fold_alphas)) if fold_alphas else 0.5
        final_epochs = int(np.median(best_epochs_list)) if best_epochs_list else MAX_EPOCHS_FINETUNE
        all_idx = np.arange(len(D.y))
        print(f"\n  Training final LSTM ensemble ({SEEDS} seeds, {final_epochs} epochs) ...")
        final_models = [
            _fit_lstm(D, all_idx, None, seed=s,
                      fixed_epochs=final_epochs,
                      lr=LR_FINETUNE,
                      pretrained_state=pretrained_state)[0]
            for s in range(SEEDS)
        ]
        final_ridge = Ridge(alpha=1.0).fit(_flat_ridge(D, all_idx), D.y[all_idx])

        pt_path = MODELS_SAVED / "mimic_lstm_forecaster.pt"
        torch.save(
            {
                "states": [m.state_dict() for m in final_models],
                "ridge_coef":      final_ridge.coef_.tolist(),
                "ridge_intercept": float(final_ridge.intercept_),
                "blend_alpha":     mean_alpha,
                "config": {
                    "mu":         D.mu,
                    "sd":         D.sd,
                    "hidden":     HIDDEN,
                    "num_layers": NUM_LAYERS,
                    "dropout":    DROPOUT,
                    "use_time":   USE_TIME,
                    "window":     WINDOW,
                },
            },
            pt_path,
        )
        print(f"  Saved LSTM+Ridge blend (alpha={mean_alpha:.2f}) -> {pt_path}")
        reg_name = "mimic_glucose_forecaster_lstm"
        mae_val  = scored[winner]
        notes    = (
            f"LSTM v3 pretrain+finetune+blend, window={WINDOW}, hidden={HIDDEN}, "
            f"layers={NUM_LAYERS}, dropout={DROPOUT}, seeds={SEEDS}, "
            f"alpha={mean_alpha:.2f}, syn_patients=1000, real_patients=30, "
            f"mean abs error {mae_val:.1f} mg/dL (pooled LOPO-CV on real data)"
        )
    else:
        all_idx = np.arange(len(D.y))
        final_ridge = Ridge(alpha=1.0).fit(_flat_ridge(D, all_idx), D.y[all_idx])
        pkl_path = MODELS_SAVED / "mimic_glucose_forecaster.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(final_ridge, fh)
        print(f"  Saved Ridge model -> {pkl_path}")
        reg_name = "mimic_glucose_forecaster"
        mae_val  = scored["Ridge"]
        notes    = f"Ridge, mean abs error {mae_val:.1f} mg/dL (pooled), LOPO-CV"

    version_id = register_model_version(
        model_name=reg_name,
        n_original_rows=int(real.sum()), n_app_rows=0,
        val_auc=0.0, val_f1=0.0, notes=notes,
    )
    promote_model(version_id, reg_name)
    print(f"  [OK] Active Track B model -> {reg_name}  (registry v{version_id})")
    return winner, mae_val


# =============================================================================
# Entry point
# =============================================================================

def main():
    init_db()
    print("=" * 60)
    print("Carepath -- unified training run")
    print("=" * 60)
    if QUICK:
        print("(--quick mode: 1 LSTM seed, 30 max epochs)")

    a_name, a_auc, a_f1 = train_track_a()
    b_name, b_mae       = train_track_b()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE -- summary")
    print("=" * 60)
    print(f"  Track A active model : {a_name}")
    print(f"                         AUC={round(a_auc,3)}  F1={round(a_f1,3)}")
    if b_name:
        print(f"  Track B active model : {b_name}")
        print(f"                         pooled LOPO MAE={b_mae:.1f} mg/dL")
    else:
        print("  Track B              : skipped (MIMIC trajectories not found)")
    print()
    print("Both models are now served by:  python -m src.api_server")
    print("  /api/predict  ->  Track A (complication risk)")
    print("  /api/forecast ->  Track B (glucose forecast, LSTM first / Ridge fallback)")


if __name__ == "__main__":
    if not TORCH_OK:
        print("WARNING: torch not installed -- Track B LSTM will be skipped.\n"
              "         Install with: pip install torch\n")
    main()
