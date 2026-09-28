"""
Feature engineering for NHANES (Track A) -- turns clean raw values into the
physiologically-grounded features the literature (Naveed et al. 2025, our
Track A base paper context) shows measurably improve prediction over raw
values alone.

Input:  data/model_ready/nhanes_clean.csv (output of preprocess_nhanes.py)
Output: data/model_ready/nhanes_features.csv

FEATURES TO ENGINEER (not yet implemented -- see TODOs)
    1. TG/HDL ratio       = LBXTLG / LBDHDD
    2. LDL/HDL ratio       = LBDLDL / LBDHDD
    3. TC/HDL ratio         = LBXTC / LBDHDD
    4. Obesity flag         = BMXBMI >= 30
    5. Prediabetes flag     = LBXGH between 5.7 and 6.4 (inclusive), OR
                              LBXGLU between 100 and 125 (inclusive)
    6. Hypertension flag from labs = BPXOSY1 >= 130 OR BPXODI1 >= 80
       (separate from the self-reported BPQ020 -- useful to compare the two)

All ratios should guard against division by zero / NaN HDL values.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_MODEL_READY

IN_PATH = DATA_MODEL_READY / "nhanes_clean.csv"
OUT_PATH = DATA_MODEL_READY / "nhanes_features.csv"


def add_lipid_ratios(df):
    hdl = df["LBDHDD"]
    trig = df["LBXTLG"]
    ldl = df["LBDLDL"]
    tc = df["LBXTC"]

    valid_hdl = hdl.notna() & (hdl != 0)
    df["TG_HDL_RATIO"] = pd.NA
    df["LDL_HDL_RATIO"] = pd.NA
    df["TC_HDL_RATIO"] = pd.NA

    df.loc[valid_hdl & trig.notna(), "TG_HDL_RATIO"] = trig[valid_hdl & trig.notna()] / hdl[valid_hdl & trig.notna()]
    df.loc[valid_hdl & ldl.notna(), "LDL_HDL_RATIO"] = ldl[valid_hdl & ldl.notna()] / hdl[valid_hdl & ldl.notna()]
    df.loc[valid_hdl & tc.notna(), "TC_HDL_RATIO"] = tc[valid_hdl & tc.notna()] / hdl[valid_hdl & tc.notna()]
    return df


def add_clinical_flags(df):
    df["OBESITY_FLAG"] = (df["BMXBMI"] >= 30).astype("Int64")
    df["PREDIABETES_FLAG"] = (
        df["LBXGH"].between(5.7, 6.4) | df["LBXGLU"].between(100, 125)
    ).astype("Int64")
    df["LAB_HYPERTENSION_FLAG"] = (
        (df["BPXOSY1"] >= 130) | (df["BPXODI1"] >= 80)
    ).astype("Int64")
    return df


def main():
    df = pd.read_csv(IN_PATH)
    df = add_lipid_ratios(df)
    df = add_clinical_flags(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows, {len(df.columns)} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
