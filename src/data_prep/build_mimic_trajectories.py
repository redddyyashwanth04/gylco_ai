"""
Build the Track B (longitudinal/progression) dataset from MIMIC-IV hosp files.

Produces two outputs:
  1. mimic_trajectories.csv     -- long format: one row per glucose/HbA1c
                                    reading per patient, ordered by time.
                                    Used to train the LSTM/CNN-LSTM sequence model.
  2. mimic_patient_summary.csv  -- wide format: one row per diabetic patient,
                                    with trend features and complication flags.
                                    Used for quick baseline models / cohort stats.

Usage:
    Put patients.csv.gz, admissions.csv.gz, diagnoses_icd.csv.gz,
    d_icd_diagnoses.csv.gz, labevents.csv.gz, d_labitems.csv.gz in RAW_DIR,
    then run: python build_mimic_trajectories.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(".")
OUT_TRAJ = Path("mimic_trajectories.csv")
OUT_SUMMARY = Path("mimic_patient_summary.csv")

# glucose (chemistry) and HbA1c item IDs, confirmed from d_labitems
GLUCOSE_ITEMID = 50931
HBA1C_ITEMID = 50852
TARGET_ITEMS = {GLUCOSE_ITEMID: "glucose", HBA1C_ITEMID: "hba1c"}

# complication ICD prefixes (ICD-9 / ICD-10)
COMPLICATIONS = {
    "nephropathy": {"icd9": ("580", "581", "582", "583", "584", "585", "586", "587", "588", "589"),
                     "icd10": ("N18",)},
    "cardiovascular": {"icd9": ("410", "411", "412", "413", "414", "428"),
                        "icd10": ("I21", "I50", "I25")},
    "neuropathy": {"icd9": ("3572",),
                   "icd10": ("G632", "E114")},
    "retinopathy": {"icd9": ("3620",),
                     "icd10": ("H360",)},
}


def is_diabetes(row):
    if row["icd_version"] == 9:
        return str(row["icd_code"]).startswith("250")
    if row["icd_version"] == 10:
        return str(row["icd_code"])[:3] in ("E08", "E09", "E10", "E11", "E12", "E13")
    return False


def flag_complication(icd_code, icd_version, prefixes):
    key = "icd9" if icd_version == 9 else "icd10"
    return str(icd_code).startswith(tuple(prefixes[key]))


def main():
    patients = pd.read_csv(RAW_DIR / "patients.csv.gz")
    admissions = pd.read_csv(RAW_DIR / "admissions.csv.gz", parse_dates=["admittime"])
    diagnoses = pd.read_csv(RAW_DIR / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    labevents = pd.read_csv(RAW_DIR / "labevents.csv.gz", parse_dates=["charttime"])

    # 1. build diabetic cohort
    diagnoses["is_dm"] = diagnoses.apply(is_diabetes, axis=1)
    dm_subjects = diagnoses.loc[diagnoses["is_dm"], "subject_id"].unique()
    print(f"Diabetic patients identified: {len(dm_subjects)}")

    # 2. complication flags per patient (+ earliest admission date each complication appears)
    diag_dm = diagnoses[diagnoses["subject_id"].isin(dm_subjects)].merge(
        admissions[["hadm_id", "admittime"]], on="hadm_id", how="left"
    )
    comp_records = []
    for _, r in diag_dm.iterrows():
        for comp_name, prefixes in COMPLICATIONS.items():
            if flag_complication(r["icd_code"], r["icd_version"], prefixes):
                comp_records.append({"subject_id": r["subject_id"], "complication": comp_name,
                                      "date": r["admittime"]})
    comp_df = pd.DataFrame(comp_records)
    first_complication = (
        comp_df.groupby(["subject_id", "complication"])["date"].min().unstack()
        if not comp_df.empty else pd.DataFrame(index=dm_subjects)
    )
    has_complication = first_complication.notna()

    # 3. lab trajectories (long format) -- this is the sequence model input
    traj = labevents[
        labevents["subject_id"].isin(dm_subjects) & labevents["itemid"].isin(TARGET_ITEMS)
    ].copy()
    traj["lab_name"] = traj["itemid"].map(TARGET_ITEMS)
    traj = traj[["subject_id", "hadm_id", "charttime", "lab_name", "valuenum"]].sort_values(
        ["subject_id", "charttime"]
    )
    traj.to_csv(OUT_TRAJ, index=False)
    print(f"Trajectory rows written: {len(traj)} -> {OUT_TRAJ}")

    # 4. per-patient summary (wide format) -- baseline / cohort-stats table
    summary_rows = []
    n_admissions = admissions[admissions["subject_id"].isin(dm_subjects)].groupby("subject_id")["hadm_id"].nunique()
    for sid in dm_subjects:
        hba1c = traj[(traj["subject_id"] == sid) & (traj["lab_name"] == "hba1c")].sort_values("charttime")
        glucose = traj[(traj["subject_id"] == sid) & (traj["lab_name"] == "glucose")].sort_values("charttime")
        row = {
            "subject_id": sid,
            "n_admissions": n_admissions.get(sid, 0),
            "n_hba1c_readings": len(hba1c),
            "first_hba1c": hba1c["valuenum"].iloc[0] if len(hba1c) else None,
            "last_hba1c": hba1c["valuenum"].iloc[-1] if len(hba1c) else None,
            "hba1c_trend": (hba1c["valuenum"].iloc[-1] - hba1c["valuenum"].iloc[0]) if len(hba1c) >= 2 else None,
            "n_glucose_readings": len(glucose),
            "mean_glucose": glucose["valuenum"].mean() if len(glucose) else None,
            "follow_up_days": (
                (hba1c["charttime"].iloc[-1] - hba1c["charttime"].iloc[0]).days if len(hba1c) >= 2 else None
            ),
        }
        for comp_name in COMPLICATIONS:
            row[f"has_{comp_name}"] = bool(has_complication.get(comp_name, {}).get(sid, False))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).merge(
        patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left"
    )
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Patient summary rows written: {len(summary)} -> {OUT_SUMMARY}")
    print()
    print("Complication prevalence in diabetic cohort:")
    for comp_name in COMPLICATIONS:
        print(f"  {comp_name}: {summary[f'has_{comp_name}'].sum()} / {len(summary)}")


if __name__ == "__main__":
    main()
