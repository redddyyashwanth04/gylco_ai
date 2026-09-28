"""
Preprocessing for the NHANES (Track A) dataset.

WHAT THIS SCRIPT NEEDS TO DO (not yet implemented -- see TODOs)
    Input:  data/processed/nhanes_merged.csv (8,153 rows x 39 features, already verified)
    Output: data/model_ready/nhanes_clean.csv

STEPS (in order -- do not reorder, later steps depend on earlier ones)
    1. Drop DBQ360 entirely -- 96% missing due to a NHANES skip pattern
       (confirmed in Progress_Report_Data_Foundation.md), not worth imputing.
    2. Recode PAD680 sentinel values: 7777 and 9999 both mean "don't know" /
       "refused" in NHANES coding, NOT literal minute counts. Replace both
       with NaN before any further processing touches this column.
    3. Impute remaining missing values:
       - Iterative/Random-Forest-based imputation (sklearn.impute.IterativeImputer)
         for continuous labs (LBXGH, LBXGLU, LBXTLG, LBDLDL, etc.)
       - Mode imputation for categorical fields (RIAGENDR, RIDRETH3, etc.)
    4. Filter TRIGLY-derived rows sensibly: LBXTLG/LBDLDL are only populated
       for the fasting subsample (see FASTQX_L / PHAFSTHR, PHAFSTMN) -- decide
       whether to impute or drop non-fasting rows for lipid-dependent features,
       and document the choice, since it directly affects Track A's lipid-ratio
       features built in feature_engineering.py.
    5. Save the cleaned table to data/model_ready/nhanes_clean.csv.

DO NOT do target/label construction here -- that's build_targets.py's job.
This script only produces a clean FEATURE table.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import NHANES_MERGED, DATA_MODEL_READY

OUT_PATH = DATA_MODEL_READY / "nhanes_clean.csv"

SENTINEL_VALUES = [7777, 9999]  # NHANES "don't know" / "refused" codes
DROP_COLUMNS = ["DBQ360"]        # 96% missing, skip-pattern artifact


def load_raw():
    return pd.read_csv(NHANES_MERGED)


def drop_unusable_columns(df):
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


def recode_sentinels(df, columns=("PAD680",)):
    import numpy as np
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(SENTINEL_VALUES, np.nan)
    return df


def impute(df):
    id_col = "SEQN"
    categorical = ["RIAGENDR", "RIDRETH3"]
    from sklearn.impute import SimpleImputer

    other_cols = [c for c in df.columns if c not in [id_col] + categorical]
    num_imputer = SimpleImputer(strategy="median")
    df[other_cols] = num_imputer.fit_transform(df[other_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    present_cat = [c for c in categorical if c in df.columns]
    if present_cat:
        df[present_cat] = cat_imputer.fit_transform(df[present_cat])

    return df


def main():
    df = load_raw()
    df = drop_unusable_columns(df)
    df = recode_sentinels(df)
    df = impute(df)
    DATA_MODEL_READY.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
