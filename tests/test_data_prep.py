"""
Basic tests for the data preparation pipeline. Run with: pytest tests/

These check STRUCTURE (files exist, columns present, no leakage) rather
than exact values, since the underlying data is real patient data that
will change as more is collected -- these tests should stay green as the
project develops, not need constant updating for new patient counts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import NHANES_MERGED, MIMIC_TRAJECTORIES, MIMIC_PATIENT_SUMMARY


def test_nhanes_merged_exists_and_has_expected_columns():
    assert NHANES_MERGED.exists(), f"Run src/data_prep/merge_nhanes_core.py first -- {NHANES_MERGED} not found"
    df = pd.read_csv(NHANES_MERGED)
    required = ["SEQN", "DIQ010", "LBXGH", "BMXBMI"]
    for col in required:
        assert col in df.columns, f"Expected column {col} missing from nhanes_merged.csv"


def test_nhanes_seqn_is_unique():
    """SEQN is the patient ID -- must be unique, or every downstream join is broken."""
    df = pd.read_csv(NHANES_MERGED)
    assert df["SEQN"].is_unique, "Duplicate SEQN values found -- patient-level join key is broken"


def test_mimic_trajectories_no_orphaned_patients():
    """Every subject_id in the trajectory table should also exist in the summary table."""
    traj = pd.read_csv(MIMIC_TRAJECTORIES)
    summary = pd.read_csv(MIMIC_PATIENT_SUMMARY)
    traj_ids = set(traj["subject_id"].unique())
    summary_ids = set(summary["subject_id"].unique())
    orphaned = traj_ids - summary_ids
    assert not orphaned, f"subject_ids in trajectories but not in summary: {orphaned}"


def test_mimic_trajectories_sorted_within_patient():
    """Each patient's readings should be usable in time order -- catches a common bug early."""
    traj = pd.read_csv(MIMIC_TRAJECTORIES, parse_dates=["charttime"])
    for subject_id, group in traj.groupby("subject_id"):
        assert group["charttime"].is_monotonic_increasing or len(group) <= 1, (
            f"Patient {subject_id}'s readings are not in chronological order -- "
            f"sort by charttime before this patient's data reaches a sequence model"
        )


def test_add_engineered_features_handles_missing_lipid_fields():
    """Partial clinic input should not crash when some lipid fields are absent."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models.predict import add_engineered_features

    patient = {
        "LBDHDD": 42,
        "LBXTC": 210,
        # LBXTLG and LBDLDL intentionally missing to simulate partial input
        "BPXOSY1": 142,
        "BPXODI1": 88,
    }

    engineered = add_engineered_features(patient)

    assert engineered["PREDIABETES_FLAG"] == 0
    assert engineered["LAB_HYPERTENSION_FLAG"] == 1
    assert "TG_HDL_RATIO" not in engineered
    assert "LDL_HDL_RATIO" not in engineered
    assert engineered["TC_HDL_RATIO"] == 5.0


def test_get_active_model_version_returns_dict_like_row():
    """The app expects model rows to behave like mappings when it reads version metadata."""
    from src.storage.db import init_db, register_model_version, get_active_model_version, promote_model, normalize_row

    init_db()
    version_id = register_model_version("demo_model", 100, 5, 0.82, 0.75, notes="demo")
    promote_model(version_id, "demo_model")
    active = get_active_model_version("demo_model")

    assert active is not None
    assert active["version_id"] == version_id
    assert active["model_name"] == "demo_model"
    assert active["val_auc"] == 0.82

    legacy = normalize_row((1, "demo_model", "2024-01-01", 100, 5, 0.82, 0.75, 1, "demo"), [
        "version_id", "model_name", "trained_at", "n_original_rows", "n_app_rows",
        "val_auc", "val_f1", "is_active", "notes"
    ])
    assert legacy["version_id"] == 1
    assert legacy["model_name"] == "demo_model"


def test_generate_clinical_recommendations_highlights_actionable_follow_up():
    """The clinician workflow should produce a concrete follow-up plan for high-risk patients."""
    from src.models.predict import generate_clinical_recommendations

    values = {
        "RIDAGEYR": 58,
        "BMXBMI": 32.0,
        "BPXOSY1": 145,
        "BPXODI1": 92,
        "LBXGH": 7.4,
        "LBXGLU": 160,
        "LBDHDD": 38,
        "LBXTC": 230,
        "LBXTLG": 210,
        "LBDLDL": 150,
    }

    recommendations = generate_clinical_recommendations(values, {
        "hypertension": 0.62,
        "nephropathy": 0.70,
        "cardiovascular": 0.80,
    })

    assert len(recommendations) >= 3
    joined = "\n".join(recommendations).lower()
    assert "blood pressure" in joined or "hypertension" in joined
    assert "lifestyle" in joined or "follow-up" in joined
    assert "risk" in joined or "monitor" in joined


def test_compute_risk_delta_handles_missing_prior_visit():
    """A new patient or sparse history should not crash when comparing current risk to a prior visit."""
    from src.models.predict import compute_risk_delta

    delta, direction = compute_risk_delta({"hypertension": 0.60}, None)
    assert delta == 0.60
    assert direction == "new-patient"

    delta, direction = compute_risk_delta({"hypertension": 0.60}, {"hypertension": 0.40})
    assert delta == 0.20
    assert direction == "up"

    delta, direction = compute_risk_delta({"hypertension": 0.60}, {"hypertension": 0.65})
    assert delta == -0.05
    assert direction == "down"
