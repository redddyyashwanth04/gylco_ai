"""
Model Transparency page -- shows the full model registry, a side-by-side
comparison of Logistic Regression vs Random Forest vs Gradient Boosting
(replicating the paper's comparison table), and the SHAP stability result
from Week 4 stress-testing.

WHY THIS PAGE EXISTS
    The retraining pipeline (src/models/retrain_pipeline.py) can promote a
    new model version automatically once it passes the validation gate. A
    clinical tool that silently changes its own behavior with no visibility
    is worse than one that doesn't retrain at all. This page is what makes
    the feedback loop trustworthy instead of a black box -- see
    Tech_Stack_Methodology_Architecture.md Section 1.6.
"""

import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_MODEL_READY, MODELS_SAVED
from src.storage.db import get_connection

st.set_page_config(page_title="Model Transparency", page_icon="\U0001F50D")
st.title("Model Transparency")
st.caption("Active model artifacts, validation records, and the paper's logistic vs RF vs boosting comparison.")

# ── Registry table ─────────────────────────────────────────────────────────────
st.subheader("Model registry")
conn = get_connection()
rows = conn.execute(
    """SELECT model_name, version_id, trained_at, n_original_rows, n_app_rows,
              val_auc, val_f1, is_active, notes
       FROM model_registry ORDER BY model_name, trained_at DESC"""
).fetchall()
conn.close()

if not rows:
    st.info(
        "No model versions registered yet. This page will populate once "
        "src/models/retrain_pipeline.py's register_model_version() has been "
        "called by a real training run."
    )
else:
    for model_name, version_id, trained_at, n_orig, n_app, val_auc, val_f1, is_active, notes in rows:
        badge = "\U0001F7E2 ACTIVE" if is_active else "inactive"
        with st.container(border=True):
            st.write(f"**{model_name}** -- version {version_id} -- {badge}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Validation AUC", f"{val_auc:.3f}" if val_auc else "N/A")
            col2.metric("Validation F1", f"{val_f1:.3f}" if val_f1 else "N/A")
            col3.metric("Original rows", n_orig)
            col4.metric("Real app-collected rows", n_app)
            st.caption(f"Trained: {trained_at}")
            if notes:
                st.caption(notes)

# ── Paper comparison: Logistic Regression vs RF vs Gradient Boosting ──────────
st.subheader("Track A: model comparison (paper Table 1)")
st.caption(
    "The paper trains three models on the same NHANES data split and compares them directly. "
    "Metrics are computed here from the saved artifacts on the held-out test set so they are "
    "reproducible from the repo without re-running training."
)

COMPARISON_MODELS = {
    "Logistic Regression (baseline)": "nhanes_traditional_baseline",
    "Random Forest (active)":         "nhanes_random_forest",
    "Gradient Boosting":              "nhanes_gradient_boosting",
}
TARGETS = ["hypertension", "nephropathy", "cardiovascular"]


@st.cache_data(show_spinner="Computing comparison metrics…")
def compute_comparison_metrics():
    """Load each saved model and evaluate on the held-out test portion of nhanes_model_ready.csv."""
    data_path = DATA_MODEL_READY / "nhanes_model_ready.csv"
    feature_path = MODELS_SAVED / "nhanes_feature_columns.pkl"
    if not data_path.exists() or not feature_path.exists():
        return None

    df = pd.read_csv(data_path)
    with open(feature_path, "rb") as fh:
        feature_cols = pickle.load(fh)

    # Use the last 20% of rows as a stable held-out set (same split convention used at training)
    split = int(len(df) * 0.8)
    test = df.iloc[split:].copy()
    X_test = test[feature_cols].fillna(test[feature_cols].median())

    results = {}
    for label, artifact in COMPARISON_MODELS.items():
        pkl_path = MODELS_SAVED / f"{artifact}.pkl"
        if not pkl_path.exists():
            results[label] = None
            continue
        with open(pkl_path, "rb") as fh:
            model = pickle.load(fh)

        aucs, f1s, accs = [], [], []
        for i, target in enumerate(TARGETS):
            if target not in test.columns:
                continue
            y = test[target].dropna()
            if len(y) < 10 or y.nunique() < 2:
                continue
            idx = y.index
            try:
                from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
                proba = model.estimators_[i].predict_proba(X_test.loc[idx])[:, 1]
                pred  = (proba >= 0.5).astype(int)
                aucs.append(roc_auc_score(y, proba))
                f1s.append(f1_score(y, pred, zero_division=0))
                accs.append(accuracy_score(y, pred))
            except Exception:
                continue

        if aucs:
            results[label] = {
                "AUC (avg)":      round(float(np.mean(aucs)), 3),
                "F1 (avg)":       round(float(np.mean(f1s)), 3),
                "Accuracy (avg)": round(float(np.mean(accs)), 3),
                "per_target": {
                    t: {
                        "AUC": round(float(a), 3),
                        "F1":  round(float(f), 3),
                    }
                    for t, a, f in zip(TARGETS, aucs, f1s)
                },
            }
        else:
            results[label] = None
    return results


metrics = compute_comparison_metrics()

if metrics is None:
    st.warning(
        "Comparison metrics require `data/model_ready/nhanes_model_ready.csv` and "
        "`models_saved/nhanes_feature_columns.pkl`. Run the data-prep and training "
        "pipeline first."
    )
elif all(v is None for v in metrics.values()):
    st.warning("None of the three saved model artifacts were found in models_saved/.")
else:
    # Summary table
    rows_display = []
    for label, m in metrics.items():
        if m:
            rows_display.append({
                "Model": label,
                "AUC (avg)": m["AUC (avg)"],
                "F1 (avg)": m["F1 (avg)"],
                "Accuracy (avg)": m["Accuracy (avg)"],
            })
    if rows_display:
        comparison_df = pd.DataFrame(rows_display).set_index("Model")
        st.dataframe(comparison_df, use_container_width=True)

    # Per-complication breakdown
    with st.expander("Per-complication breakdown", expanded=False):
        for label, m in metrics.items():
            if m is None:
                st.write(f"**{label}** — artifact not found")
                continue
            st.write(f"**{label}**")
            per_cols = st.columns(len(TARGETS))
            for idx, target in enumerate(TARGETS):
                vals = m["per_target"].get(target, {})
                with per_cols[idx]:
                    st.metric(
                        target.capitalize(),
                        f"AUC {vals.get('AUC', 'N/A')}",
                        delta=f"F1 {vals.get('F1', 'N/A')}",
                        delta_color="off",
                    )

    st.caption(
        "Metrics computed on the held-out 20% of NHANES rows. "
        "The Random Forest is the active model served by `/api/predict`."
    )

# ── SHAP stability (Week 4) ────────────────────────────────────────────────────
st.subheader("SHAP explanation stability (Week 4 stress-test)")
st.caption(
    "Explanation stability answers: if we nudge each input slightly, does the "
    "SHAP ranking stay the same? A model whose top-5 feature order flips under "
    "small noise is not trustworthy as a clinical explainer."
)


@st.cache_data(show_spinner="Running SHAP stability check…")
def compute_shap_stability():
    """
    Runs the perturbation test from Week 4: add ±5 % Gaussian noise to each
    feature 20 times and measure how often the top-3 SHAP feature *set*
    (not order) stays identical to the unperturbed result.
    Returns a dict {target: stability_pct} and an overall average.
    """
    data_path = DATA_MODEL_READY / "nhanes_model_ready.csv"
    feature_path = MODELS_SAVED / "nhanes_feature_columns.pkl"
    rf_path = MODELS_SAVED / "nhanes_random_forest.pkl"

    if not all(p.exists() for p in [data_path, feature_path, rf_path]):
        return None

    try:
        import shap
    except ImportError:
        return {"error": "shap not installed — run pip install shap"}

    df = pd.read_csv(data_path)
    with open(feature_path, "rb") as fh:
        feature_cols = pickle.load(fh)
    with open(rf_path, "rb") as fh:
        rf = pickle.load(fh)

    medians = df[feature_cols].median()
    # Use one representative patient (row 0 of the test set)
    split = int(len(df) * 0.8)
    test = df.iloc[split:split + 1][feature_cols].fillna(medians)

    N_PERTURB = 20
    NOISE_FRAC = 0.05
    TOP_K = 3

    stability = {}
    for i, target in enumerate(TARGETS):
        estimator = rf.estimators_[i]
        try:
            explainer = shap.TreeExplainer(estimator)

            def top_k_set(x_row):
                sv = explainer.shap_values(x_row)
                arr = sv[1][0] if isinstance(sv, list) else sv[0]
                return frozenset(np.argsort(-np.abs(arr))[:TOP_K])

            baseline_top = top_k_set(test)
            matches = 0
            rng = np.random.default_rng(seed=42)
            for _ in range(N_PERTURB):
                noise = rng.normal(0, NOISE_FRAC, size=test.shape)
                perturbed = test * (1 + noise)
                if top_k_set(perturbed) == baseline_top:
                    matches += 1
            stability[target] = round(matches / N_PERTURB * 100, 1)
        except Exception as exc:
            stability[target] = f"error: {exc}"

    return stability


shap_stability = compute_shap_stability()

if shap_stability is None:
    st.info(
        "SHAP stability requires `models_saved/nhanes_random_forest.pkl` and the "
        "prepared training data. Run the training pipeline first."
    )
elif "error" in shap_stability:
    st.warning(shap_stability["error"])
else:
    st.write(
        "Stability = % of 20 runs (±5 % Gaussian noise) where the top-3 SHAP "
        "feature **set** is unchanged from the unperturbed prediction."
    )
    stab_cols = st.columns(len(TARGETS))
    overall_vals = [v for v in shap_stability.values() if isinstance(v, float)]
    for idx, target in enumerate(TARGETS):
        val = shap_stability.get(target, "N/A")
        display = f"{val}%" if isinstance(val, float) else val
        stab_cols[idx].metric(f"{target.capitalize()} stability", display)
    if overall_vals:
        avg_stability = round(float(np.mean(overall_vals)), 1)
        st.metric("Overall stability (avg across complications)", f"{avg_stability}%")
        st.caption(
            f"Week 4 result: top-3 SHAP drivers are consistent across {avg_stability}% of "
            "perturbation runs. Values above 80% are considered stable for clinical decision support."
        )
