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


def load_active_model():
    name = (MODELS_SAVED / "nhanes_active_name.txt").read_text().strip()
    with open(MODELS_SAVED / f"{name}.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_SAVED / "nhanes_feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return name, model, feature_cols


def add_engineered_features(p):
    """Same feature engineering used in training -- must stay in sync with feature_engineering.py."""
    p = dict(p)
    hdl = p.get("LBDHDD")
    if hdl and hdl != 0:
        triglycerides = p.get("LBXTLG")
        ldl = p.get("LBDLDL")
        total_chol = p.get("LBXTC")

        if triglycerides is not None:
            p["TG_HDL_RATIO"] = triglycerides / hdl
        if ldl is not None:
            p["LDL_HDL_RATIO"] = ldl / hdl
        if total_chol is not None:
            p["TC_HDL_RATIO"] = total_chol / hdl

    p["PREDIABETES_FLAG"] = int(5.7 <= p.get("LBXGH", 0) <= 6.4 or 100 <= p.get("LBXGLU", 0) <= 125)
    p["LAB_HYPERTENSION_FLAG"] = int(p.get("BPXOSY1", 0) >= 130 or p.get("BPXODI1", 0) >= 80)
    return p


def predict_patient(patient_values):
    """
    patient_values: dict of raw NHANES-style values (any missing ones are
    filled with the training-set typical value, so partial input still works).
    Returns: dict like {"hypertension": 0.62, "nephropathy": 0.18, ...}
    """
    name, model, feature_cols = load_active_model()
    row = add_engineered_features(patient_values)
    x = pd.DataFrame([row]).reindex(columns=feature_cols)

    # fill anything the clinician didn't enter with the training medians
    train = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "data" / "model_ready" / "nhanes_model_ready.csv")
    x = x.fillna(train[feature_cols].median())

    probas = model.predict_proba(x)  # list of arrays, one per target
    return {t: float(probas[i][0, 1]) for i, t in enumerate(TARGETS)}, name, x


def compute_risk_delta(current_risks, previous_risks):
    """Return a risk delta and human-readable trend direction for doctor-facing comparisons."""
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
    """Doctor-facing follow-up suggestions based on current risk and the clinician's values."""
    recs = []
    bmi = patient_values.get("BMXBMI") or 0
    hba1c = patient_values.get("LBXGH") or 0
    glucose = patient_values.get("LBXGLU") or 0
    systolic = patient_values.get("BPXOSY1") or 0
    diastolic = patient_values.get("BPXODI1") or 0

    if risks.get("hypertension", 0) >= 0.5 or systolic >= 130 or diastolic >= 80:
        recs.append("Hypertension risk is elevated. Review blood pressure control, verify home readings, and confirm medication adherence before the next follow-up visit.")
    if risks.get("nephropathy", 0) >= 0.5 or hba1c >= 7 or glucose >= 140:
        recs.append("Nephropathy risk is concerning. Check albuminuria and kidney function and consider earlier renal review if the risk trend is rising.")
    if risks.get("cardiovascular", 0) >= 0.5 or bmi >= 30 or (patient_values.get("LBDHDD") and patient_values.get("LBXTC") and patient_values["LBXTC"] / patient_values["LBDHDD"] > 5):
        recs.append("Cardiovascular risk is elevated. Prioritize lipid control, consider statin suitability, and review smoking, activity, and nutrition with the patient.")
    if bmi >= 30:
        recs.append("Lifestyle intervention should be a key part of management. Reinforce weight-loss targets, activity goals, and meal-planning support.")
    if hba1c >= 7 or glucose >= 140:
        recs.append("Glycemic risk is elevated. Reassess medication adherence and schedule a follow-up in 6–12 weeks or sooner if the trend is worsening.")

    if not recs:
        recs.append("Current risk is low-to-moderate. Continue routine monitoring and repeat reassessment at the next planned clinic visit.")

    recs.append("For a real consultation, document the risk trend over time, align the plan with patient goals, and schedule follow-up based on the direction of change.")
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
