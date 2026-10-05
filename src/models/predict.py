"""
Track A prediction pipeline. Give it one patient's values, get back risk
for each complication.

Run a demo: python src/models/predict.py
Or import: from src.models.predict import predict_patient
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MODELS_SAVED

import pandas as pd

TARGETS = ["hypertension", "nephropathy", "cardiovascular"]


_MODEL_CACHE: dict = {}   # keys: name, model, feature_cols, mtime, medians


def load_active_model():
    """Load the active Track A model. Reloads automatically if the artifact changes on disk.

    Training medians are computed once here and cached so predict_patient()
    never reads the training CSV again (P2 fix: was re-reading on every request).
    """
    name_path = MODELS_SAVED / "nhanes_active_name.txt"
    name = name_path.read_text().strip()
    pkl_path = MODELS_SAVED / f"{name}.pkl"
    mtime = pkl_path.stat().st_mtime

    if _MODEL_CACHE.get("name") == name and _MODEL_CACHE.get("mtime") == mtime:
        return _MODEL_CACHE["name"], _MODEL_CACHE["model"], _MODEL_CACHE["feature_cols"]

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    with open(MODELS_SAVED / "nhanes_feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # Compute and cache medians once — avoids re-reading the 8k-row CSV on every prediction.
    train_path = Path(__file__).resolve().parent.parent.parent / "data" / "model_ready" / "nhanes_model_ready.csv"
    medians = pd.read_csv(train_path)[feature_cols].median()

    _MODEL_CACHE.update(name=name, model=model, feature_cols=feature_cols,
                        mtime=mtime, medians=medians)
    return name, model, feature_cols


def add_engineered_features(p):
    """Same feature engineering used in training -- must stay in sync with feature_engineering.py.
    None-safe: an empty optional field arrives as None and must not crash comparisons."""
    p = dict(p)
    hdl = p.get("LBDHDD")
    if hdl:
        triglycerides = p.get("LBXTLG")
        ldl = p.get("LBDLDL")
        total_chol = p.get("LBXTC")

        if triglycerides is not None:
            p["TG_HDL_RATIO"] = triglycerides / hdl
        if ldl is not None:
            p["LDL_HDL_RATIO"] = ldl / hdl
        if total_chol is not None:
            p["TC_HDL_RATIO"] = total_chol / hdl

    hba1c = p.get("LBXGH") or 0
    glucose = p.get("LBXGLU") or 0
    p["PREDIABETES_FLAG"] = int(5.7 <= hba1c <= 6.4 or 100 <= glucose <= 125)
    p["LAB_HYPERTENSION_FLAG"] = int((p.get("BPXOSY1") or 0) >= 130 or (p.get("BPXODI1") or 0) >= 80)
    return p


def predict_patient(patient_values):
    """
    patient_values: dict of raw NHANES-style values (missing ones are filled
    with training medians, so partial input still works).
    Returns: (risks dict, model name, feature frame)
    """
    name, model, feature_cols = load_active_model()
    row = add_engineered_features(patient_values)
    x = pd.DataFrame([row]).reindex(columns=feature_cols)
    x = x.fillna(_MODEL_CACHE["medians"])  # cached at model load — no CSV read per request
    probas = model.predict_proba(x)
    return {t: float(probas[i][0, 1]) for i, t in enumerate(TARGETS)}, name, x


def compute_risk_delta(current_risks, previous_risks):
    """Return a risk delta and trend direction for clinician-facing comparisons."""
    if not previous_risks:
        total = sum(current_risks.values()) / len(current_risks) if current_risks else 0.0
        return round(float(total), 4), "new-patient"

    deltas = []
    for key, value in current_risks.items():
        prev = previous_risks.get(key, value)
        deltas.append(float(value) - float(prev))

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    avg_delta = round(float(avg_delta), 4)
    if avg_delta > 0.02:
        return avg_delta, "up"
    if avg_delta < -0.02:
        return avg_delta, "down"
    return avg_delta, "stable"


def generate_clinical_recommendations(patient_values, risks):
    """
    Points for clinician review. These are considerations, not orders: the
    tool supports the clinician's decision and never makes it.
    """
    recs = []
    bmi = patient_values.get("BMXBMI") or 0
    hba1c = patient_values.get("LBXGH") or 0
    glucose = patient_values.get("LBXGLU") or 0
    systolic = patient_values.get("BPXOSY1") or 0
    diastolic = patient_values.get("BPXODI1") or 0
    hdl = patient_values.get("LBDHDD")
    total_chol = patient_values.get("LBXTC")

    if risks.get("hypertension", 0) >= 0.5 or systolic >= 130 or diastolic >= 80:
        recs.append("Hypertension risk is elevated. Consider reviewing blood pressure control, home readings, and medication adherence.")
    if risks.get("nephropathy", 0) >= 0.5:
        recs.append("Estimated nephropathy risk is elevated. Consider reviewing albuminuria and kidney function results, and whether earlier renal review is warranted if the risk trend is rising.")
    if risks.get("cardiovascular", 0) >= 0.5 or bmi >= 30 or (hdl and total_chol and total_chol / hdl > 5):
        recs.append("Cardiovascular risk markers are elevated. Consider reviewing lipid management, smoking, activity, and nutrition with the patient.")
    if bmi >= 30:
        recs.append("BMI is in the obesity range. Lifestyle support (activity, meal planning, weight goals) may be worth discussing.")
    if hba1c >= 7 or glucose >= 140:
        recs.append("Glycemic markers are above common targets. Consider reviewing adherence and the follow-up interval; timing is a clinical judgment.")

    if not recs:
        recs.append("Current estimated risk is low to moderate. Routine monitoring at the next planned visit may be appropriate.")

    recs.append("Document the risk trend across visits and align the plan with patient goals. Follow-up timing remains the clinician's decision.")
    return recs


if __name__ == "__main__":
    example = {
        "RIDAGEYR": 58, "RIAGENDR": 1, "LBXGH": 7.4, "LBXGLU": 150,
        "BMXBMI": 31.0, "BPXOSY1": 142, "BPXODI1": 88,
        "LBXTC": 210, "LBDHDD": 42, "LBXTLG": 190, "LBDLDL": 130,
    }
    risks, model_name, _ = predict_patient(example)
    print(f"Model used: {model_name}")
    for complication, risk in risks.items():
        print(f"  {complication:15s} {risk:.1%}")