"""
Merge the core NHANES August 2021-August 2023 files into one
analysis-ready dataset for the T2D progression / complications project.

Usage:
    1. Put all downloaded .XPT files in one folder, e.g. ./nhanes_raw/
    2. Update RAW_DIR below if needed.
    3. Run: python merge_nhanes_core.py
    4. Output: ./nhanes_merged.csv
"""

import pandas as pd
from functools import reduce
from pathlib import Path

RAW_DIR = Path(r"C:\nhanes_raw")   # folder where your .XPT files live
OUT_FILE = Path(r"C:\nhanes_raw\nhanes_merged.csv")

# file -> which columns to keep (SEQN is the join key, always kept automatically)
# NOTE: if a file/column name below doesn't match what you downloaded,
# open the matching .htm codebook page for that file and adjust the name.
FILES_AND_COLUMNS = {
    "DEMO_L.XPT":     ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR"],  # age, sex, ethnicity, poverty ratio
    "DIQ_L.XPT":      ["DIQ010", "DIQ050", "DIQ070"],                    # diabetes dx, insulin use, oral meds
    "GHB_L.XPT":      ["LBXGH"],                                        # HbA1c %
    "GLU_L.XPT":      ["LBXGLU"],                                       # fasting glucose
    "FASTQX_L.XPT":   ["PHAFSTHR", "PHAFSTMN"],                         # fasting hours/minutes
    "TCHOL_L.XPT":    ["LBXTC"],                                        # total cholesterol
    "HDL_L.XPT":      ["LBDHDD"],                                       # HDL cholesterol
    "TRIGLY_L.XPT":   ["LBXTR", "LBXTLG", "LBDLDL"],                    # triglycerides (either name), LDL
    "BMX_L.XPT":      ["BMXBMI", "BMXWT", "BMXHT", "BMXWAIST"],         # BMI, weight, height, waist
    "BPXO_L.XPT":     ["BPXOSY1", "BPXODI1"],                           # systolic/diastolic BP
    "BPQ_L.XPT":      ["BPQ020"],                                       # told you had hypertension
    "KIQ_U_L.XPT":    ["KIQ022"],                                       # told you had weak/failing kidneys
    "ALB_CR_L.XPT":   ["URXUMA", "URXUCR"],                             # urine albumin, urine creatinine
    "BIOPRO_L.XPT":   ["LBXSCR", "LBXSAL", "LBXSATSI", "LBXSASSI"],     # serum creatinine, albumin, liver enzymes
    "MCQ_L.XPT":      ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"],     # CHF, CHD, heart attack, stroke
    "DBQ_L.XPT":      ["DBQ360", "DBQ930"],                             # fast food frequency, food security
    "PAQ_L.XPT":      ["PAD680"],                                       # minutes sedentary activity per day
    "SMQ_L.XPT":      ["SMQ020"],                                       # smoked 100+ cigarettes in life
    "ALQ_L.XPT":      ["ALQ121"],                                       # alcohol frequency past 12 months
    "INQ_L.XPT":      ["INDFMMPI"],                                     # family income proxy (if present)
}

def load_file(fname, cols):
    path = RAW_DIR / fname
    if not path.exists():
        print(f"  [skip] {fname} not found in {RAW_DIR} -- skipping")
        return None
    df = pd.read_sas(path, format="xport")
    df.columns = [c.upper() for c in df.columns]
    keep = ["SEQN"] + [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [warn] {fname}: columns not found -> {missing} (check the .htm codebook page)")
    return df[keep]

def main():
    print(f"Reading files from: {RAW_DIR.resolve()}")
    frames = []
    for fname, cols in FILES_AND_COLUMNS.items():
        df = load_file(fname, cols)
        if df is not None:
            frames.append(df)
            print(f"  [ok] {fname}: {df.shape[0]} rows, {df.shape[1]-1} feature columns")

    if not frames:
        print("No files loaded -- check RAW_DIR path and file names.")
        return

    merged = reduce(lambda l, r: pd.merge(l, r, on="SEQN", how="outer"), frames)

    # keep adults only (18+), matching typical T2D-risk study populations
    if "RIDAGEYR" in merged.columns:
        merged = merged[merged["RIDAGEYR"] >= 18]

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nMerged dataset: {merged.shape[0]} rows x {merged.shape[1]} columns")
    print(f"Saved to: {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
