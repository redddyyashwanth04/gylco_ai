"""
One function the app calls to get: risk numbers + which features drove them
+ a plain-language sentence. Needs `pip install shap groq` to actually run
(not testable in this sandbox -- no internet here to install them).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.predict import load_active_model, add_engineered_features, TARGETS
from src.explainability.shap_explainer import explain_tree_model, top_features_for_patient
from src.explainability.llm_interpreter import generate_explanation

import pandas as pd


def explain_patient(patient_values):
    """
    patient_values: dict of raw NHANES-style values.
    Returns: {
      "risks": {"hypertension": 0.62, ...},
      "top_features": {"hypertension": {"LBXGH": 0.31, ...}, ...},
      "narrative": {"hypertension": "This patient's risk is driven by...", ...}
    }
    """
    name, model, feature_cols = load_active_model()
    row = add_engineered_features(patient_values)
    x = pd.DataFrame([row]).reindex(columns=feature_cols)
    train = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "data" / "model_ready" / "nhanes_model_ready.csv")
    x = x.fillna(train[feature_cols].median())

    probas = model.predict_proba(x)
    risks = {t: float(probas[i][0, 1]) for i, t in enumerate(TARGETS)}

    result = {"risks": risks, "top_features": {}, "narrative": {}}
    for i, target in enumerate(TARGETS):
        try:
            # each target has its own underlying estimator inside the MultiOutputClassifier
            estimator = model.estimators_[i]
            _, shap_values = explain_tree_model(estimator, x)
            top = top_features_for_patient(shap_values[1] if isinstance(shap_values, list) else shap_values,
                                            feature_cols, patient_idx=0, top_n=5)
            result["top_features"][target] = top
            result["narrative"][target] = generate_explanation(
                patient_id="current", complication_name=target,
                predicted_risk=risks[target], shap_values=top,
                feature_values={k: row.get(k) for k in top},
            )
        except Exception as e:
            result["narrative"][target] = f"[explanation unavailable: {e}]"
    return result


if __name__ == "__main__":
    example = {
        "RIDAGEYR": 58, "RIAGENDR": 1, "LBXGH": 7.4, "LBXGLU": 150,
        "BMXBMI": 31.0, "BPXOSY1": 142, "BPXODI1": 88,
        "LBXTC": 210, "LBDHDD": 42, "LBXTLG": 190, "LBDLDL": 130,
    }
    out = explain_patient(example)
    for t, r in out["risks"].items():
        print(f"{t}: {r:.1%}")
        print(f"  {out['narrative'].get(t)}")
