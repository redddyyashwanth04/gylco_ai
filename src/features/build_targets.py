"""
Builds the multi-label complication targets for Track A, matching the
structure of our base paper (Zamani et al., 2026) so results are directly
comparable.

Input:  data/model_ready/nhanes_features.csv
Output: data/model_ready/nhanes_model_ready.csv (features + target columns)

TARGET LABELS (5, matching the base paper's micro/macrovascular split)
    - hypertension   = BPQ020 == 1 (self-reported diagnosed hypertension)
    - nephropathy    = KIQ022 == 1 OR URXUMA above the microalbuminuria
                       threshold (>=30 mg/L) -- combine self-report + lab marker
    - cardiovascular = any of MCQ160B, MCQ160C, MCQ160E, MCQ160F == 1
                       (congestive heart failure, coronary heart disease,
                       heart attack, stroke)
    - obesity        = OBESITY_FLAG (from feature_engineering.py)
    - diabetes       = DIQ010 == 1 (the primary disease label, kept separate
                       from the complication labels for the traditional-model
                       baseline comparison)

Each label should end up as its own 0/1 (or pd.NA for missing) column, ready
to feed directly into sklearn's MultiOutputClassifier / ClassifierChain.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_MODEL_READY

IN_PATH = DATA_MODEL_READY / "nhanes_features.csv"
OUT_PATH = DATA_MODEL_READY / "nhanes_model_ready.csv"

TARGET_COLUMNS = ["hypertension", "nephropathy", "cardiovascular", "obesity", "diabetes"]


def build_targets(df):
    df["hypertension"] = (df["BPQ020"] == 1).astype("Int64")

    # TODO: confirm the correct microalbuminuria cutoff for URXUMA (mg/L) with
    # a clinical reference before finalizing -- 30 mg/L is the commonly cited
    # threshold but should be verified against current guidelines.
    df["nephropathy"] = (
        (df["KIQ022"] == 1) | (df["URXUMA"] >= 30)
    ).astype("Int64")

    cvd_cols = ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"]
    df["cardiovascular"] = (df[cvd_cols] == 1).any(axis=1).astype("Int64")

    df["obesity"] = df["OBESITY_FLAG"]
    df["diabetes"] = (df["DIQ010"] == 1).astype("Int64")

    return df


def main():
    df = pd.read_csv(IN_PATH)
    df = build_targets(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows with targets {TARGET_COLUMNS} to {OUT_PATH}")
    for t in TARGET_COLUMNS:
        print(f"  {t}: {df[t].sum()} positive / {df[t].notna().sum()} non-missing")


if __name__ == "__main__":
    main()
