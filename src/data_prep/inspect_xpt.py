"""
Quick helper: print the real column names inside an NHANES .XPT file.
Run this on any file where the merge script reports missing columns.

Usage:
    python inspect_xpt.py DBQ_L.XPT
    python inspect_xpt.py PAQ_L.XPT
"""

import sys
import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\nhanes_raw")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_xpt.py <filename.XPT>")
        return
    fname = sys.argv[1]
    path = RAW_DIR / fname
    if not path.exists():
        print(f"File not found: {path}")
        return
    df = pd.read_sas(path, format="xport")
    print(f"{fname}: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("Columns:")
    for c in df.columns:
        print(" ", c)

if __name__ == "__main__":
    main()
