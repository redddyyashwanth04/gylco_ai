"""
One function the app calls to get: risk numbers + top SHAP drivers per
complication + a plain-language sentence.

Needs `pip install shap` (and `groq` + GROQ_API_KEY for LLM narratives).
Without Groq it falls back to a deterministic template, so the demo works offline.
The LLM only ever receives SHAP values and feature values, never a patient ID.

v3 architecture change (SHAP stability fix):
  Track A now trains three SEPARATE single-output RandomForest classifiers — one
  per label — instead of one MultiOutputClassifier.  This removes the shared-SHAP-
  budget problem where near-equal features competed across labels and produced 62%
  top-3 stability.  Each per-label model has its own independent SHAP budget, so
  the top features are well-separated and stable.

  Correlated-feature deduplication is kept: when both LBXTC (total cholesterol)
  and TC_HDL_RATIO (total/HDL ratio) appear in the top factors for the same label,
  only the ratio is retained (more informative; SHAP is merged into it).
  Same logic for LDL_HDL_RATIO vs LBDLDL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import shap

from config import DATA_MODEL_READY
from src.models.predict import load_per_label_models, add_engineered_features, TARGETS
from src.explainability.llm_interpreter import generate_explanation, FEATURE_LABELS

# ── correlated-feature deduplication pairs ────────────────────────────────────
REDUNDANT_PAIRS = [
    ("LBXTC",  "TC_HDL_RATIO"),   # total cholesterol  →  total/HDL ratio
    ("LBDLDL", "LDL_HDL_RATIO"),  # LDL                →  LDL/HDL ratio
]

LABELS = {
    **FEATURE_LABELS,
    "TG_HDL_RATIO":          "triglyceride/HDL ratio",
    "LDL_HDL_RATIO":         "LDL/HDL ratio",
    "TC_HDL_RATIO":          "total/HDL cholesterol ratio",
    "PREDIABETES_FLAG":      "prediabetes range",
    "LAB_HYPERTENSION_FLAG": "elevated measured BP",
    "LBXTC":                 "total cholesterol (mg/dL)",
    "RIAGENDR":              "sex",
    "RIDRETH3":              "ethnicity group",
}

_TRAIN = None


def _train():
    global _TRAIN
    if _TRAIN is None:
        _TRAIN = pd.read_csv(DATA_MODEL_READY / "nhanes_model_ready.csv")
    return _TRAIN


# ── SHAP for one single-output classifier ────────────────────────────────────

def _positive_class_shap(model, x, background):
    """TreeExplainer on a single-output classifier; generic fallback if unsupported."""
    try:
        sv = shap.TreeExplainer(model).shap_values(x)
    except Exception:
        def f(d):
            return model.predict_proba(pd.DataFrame(d, columns=x.columns))[:, 1]
        sv = shap.Explainer(f, background)(x).values
    if isinstance(sv, list):
        sv = sv[1]      # take positive-class slice from list format
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv[0]        # 1-D array, shape (n_features,)


# ── correlated-feature deduplication ─────────────────────────────────────────

def _deduplicate(factors):
    """When both raw and ratio feature appear, keep ratio and merge SHAP."""
    feat_index = {f["feature"]: i for i, f in enumerate(factors)}
    to_drop = set()
    for raw_feat, ratio_feat in REDUNDANT_PAIRS:
        if raw_feat in feat_index and ratio_feat in feat_index:
            factors[feat_index[ratio_feat]]["shap"] += factors[feat_index[raw_feat]]["shap"]
            to_drop.add(raw_feat)
    return [f for f in factors if f["feature"] not in to_drop]


# ── template fallback ─────────────────────────────────────────────────────────

def _template(target, risk, factors):
    up   = [f["label"] for f in factors if f["shap"] > 0][:3]
    down = [f["label"] for f in factors if f["shap"] < 0][:2]
    s = f"Estimated {target} risk is {risk:.0%}."
    if up:
        s += " Factors raising it most: " + ", ".join(up) + "."
    if down:
        s += " Factors lowering it: " + ", ".join(down) + "."
    return s + " Decision-support estimate from a research model, not a diagnosis."


# ── public API ────────────────────────────────────────────────────────────────

def explain_patient(patient_values, use_llm=True):
    """Return risks, top-5 SHAP factors, and narrative for each complication."""
    per_label_models, cols, risks, medians = load_per_label_models(patient_values)

    train      = _train()
    background = train[cols].fillna(medians).sample(min(100, len(train)), random_state=0)

    row     = add_engineered_features(patient_values)
    entered = set(row)
    x       = pd.DataFrame([row]).reindex(columns=cols).fillna(medians)

    out = {"risks": risks, "factors": {}, "narrative": {}, "narrativeSource": {}}

    for target in TARGETS:
        try:
            model = per_label_models[target]
            sv    = _positive_class_shap(model, x, background)
            top   = np.argsort(-np.abs(sv))[:5]

            factors = [{
                "feature": cols[j],
                "label":   LABELS.get(cols[j], cols[j]),
                "shap":    float(sv[j]),
                "value":   float(x.iloc[0, j]),
                "entered": cols[j] in entered,
            } for j in top]

            factors = _deduplicate(factors)
            factors.sort(key=lambda f: -abs(f["shap"]))
            factors = factors[:5]

            text   = None
            source = "llm"
            if use_llm:
                text = generate_explanation(
                    "current", target, risks[target],
                    {f["feature"]: f["shap"]  for f in factors},
                    {f["feature"]: f["value"] for f in factors},
                )
            if not text or text.startswith("["):
                text, source = _template(target, risks[target], factors), "template"

            out["factors"][target]         = factors
            out["narrative"][target]       = text
            out["narrativeSource"][target] = source

        except Exception:
            out["factors"][target]         = []
            out["narrative"][target]       = "Explanation unavailable for this estimate."
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
