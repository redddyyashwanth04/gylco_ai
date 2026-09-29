"""
One function the app calls to get: risk numbers + top SHAP drivers per
complication + a plain-language sentence.

Needs `pip install shap` (and `groq` + GROQ_API_KEY for LLM narratives).
Without Groq it falls back to a deterministic template, so the demo works offline.
The LLM only ever receives SHAP values and feature values, never a patient ID.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import shap

from config import DATA_MODEL_READY
from src.models.predict import load_active_model, add_engineered_features, TARGETS
from src.explainability.llm_interpreter import generate_explanation, FEATURE_LABELS

LABELS = {
    **FEATURE_LABELS,
    "TG_HDL_RATIO": "triglyceride/HDL ratio",
    "LDL_HDL_RATIO": "LDL/HDL ratio",
    "TC_HDL_RATIO": "total/HDL cholesterol ratio",
    "PREDIABETES_FLAG": "prediabetes range",
    "LAB_HYPERTENSION_FLAG": "elevated measured BP",
    "LBXTC": "total cholesterol (mg/dL)",
    "RIAGENDR": "sex",
    "RIDRETH3": "ethnicity group",
}
_TRAIN = None


def _train():
    global _TRAIN
    if _TRAIN is None:
        _TRAIN = pd.read_csv(DATA_MODEL_READY / "nhanes_model_ready.csv")
    return _TRAIN


def _positive_class_shap(estimator, x, background):
    """TreeExplainer for RF/LightGBM; generic fallback for models it can't handle (e.g. HistGradientBoosting)."""
    try:
        sv = shap.TreeExplainer(estimator).shap_values(x)
    except Exception:
        def f(d):
            return estimator.predict_proba(pd.DataFrame(d, columns=x.columns))[:, 1]
        sv = shap.Explainer(f, background)(x).values
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def _template(target, risk, factors):
    up = [f["label"] for f in factors if f["shap"] > 0][:3]
    down = [f["label"] for f in factors if f["shap"] < 0][:2]
    s = f"Estimated {target} risk is {risk:.0%}."
    if up:
        s += " Factors raising it most: " + ", ".join(up) + "."
    if down:
        s += " Factors lowering it: " + ", ".join(down) + "."
    return s + " Decision-support estimate from a research model, not a diagnosis."


def explain_patient(patient_values, use_llm=True):
    name, model, cols = load_active_model()
    train = _train()
    medians = train[cols].median()
    row = add_engineered_features(patient_values)
    entered = set(row)
    x = pd.DataFrame([row]).reindex(columns=cols).fillna(medians)
    background = train[cols].fillna(medians).sample(min(100, len(train)), random_state=0)

    probas = model.predict_proba(x)
    risks = {t: float(probas[i][0, 1]) for i, t in enumerate(TARGETS)}

    out = {"risks": risks, "factors": {}, "narrative": {}, "narrativeSource": {}}
    for i, target in enumerate(TARGETS):
        try:
            sv = _positive_class_shap(model.estimators_[i], x, background)[0]
            top = np.argsort(-np.abs(sv))[:5]
            factors = [{
                "feature": cols[j],
                "label": LABELS.get(cols[j], cols[j]),
                "shap": float(sv[j]),
                "value": float(x.iloc[0, j]),
                "entered": cols[j] in entered,
            } for j in top]

            text = None
            if use_llm:
                text = generate_explanation(
                    "current", target, risks[target],
                    {f["feature"]: f["shap"] for f in factors},
                    {f["feature"]: f["value"] for f in factors},
                )
            source = "llm"
            if not text or text.startswith("["):
                text, source = _template(target, risks[target], factors), "template"

            out["factors"][target] = factors
            out["narrative"][target] = text
            out["narrativeSource"][target] = source
        except Exception:
            out["factors"][target] = []
            out["narrative"][target] = "Explanation unavailable for this estimate."
            out["narrativeSource"][target] = "none"
    return out


if __name__ == "__main__":
    example = {
        "RIDAGEYR": 58, "RIAGENDR": 1, "LBXGH": 7.4, "LBXGLU": 150,
        "BMXBMI": 31.0, "BPXOSY1": 142, "BPXODI1": 88,
        "LBXTC": 210, "LBDHDD": 42, "LBXTLG": 190, "LBDLDL": 130,
    }
    res = explain_patient(example)
    for t, r in res["risks"].items():
        print(f"{t}: {r:.1%}\n  {res['narrative'][t]}")