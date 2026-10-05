"""
Track B training: forecast the next glucose reading from the previous 4.

Run: python src/models/train_track_b.py

How it works (short):
  1. Slice each patient's real glucose readings into windows of 5.
  2. Training uses real windows + jittered copies (augmentation).
  3. Testing uses REAL windows only, from a patient the model never saw
     (leave-one-patient-out) -- so augmented copies can never leak.
  4. First 4 readings = input, 5th reading = what we predict.
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MIMIC_TRAJECTORIES, MODELS_SAVED
from src.features.augmentation import jitter, window_slice
from src.storage.db import init_db, register_model_version, promote_model

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut

WINDOW = 5
STRIDE = 3
JITTER_COPIES = 4


def make_features(w):
    """w: array of 4 previous readings -> feature row."""
    w = np.asarray(w, dtype=float)
    return [*w, w.mean(), w.std(), w[-1] - w[0]]


def build_windows(traj_df):
    """Returns X, y, patient_ids, is_augmented (one row per window)."""
    glucose = traj_df[traj_df["lab_name"] == "glucose"]
    X, y, groups, aug = [], [], [], []
    for pid, g in glucose.groupby("subject_id"):
        values = g.sort_values("charttime")["valuenum"].tolist()
        for win in window_slice(values, min_window=WINDOW, stride=STRIDE):
            if len(win) < WINDOW:
                continue  # patient has too few readings to make a window
            X.append(make_features(win[:-1])); y.append(win[-1]); groups.append(pid); aug.append(False)
            for copy in jitter(win, n_copies=JITTER_COPIES, seed=1):
                X.append(make_features(copy[:-1])); y.append(copy[-1]); groups.append(pid); aug.append(True)
    return np.array(X), np.array(y), np.array(groups), np.array(aug)


MODELS = {
    "Naive (repeat last value)": None,  # no training, just copy the last reading
    "Linear (Ridge)": lambda: Ridge(alpha=1.0),
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=42),
}


def main():
    traj = pd.read_csv(MIMIC_TRAJECTORIES)
    X, y, groups, aug = build_windows(traj)
    n_patients = len(set(groups))
    n_real = int((~aug).sum())
    print(f"Real patients used: {n_patients} | real windows: {n_real} | total examples (with augmentation): {len(y)}")

    errors = {name: [] for name in MODELS}
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups):
        test_idx = test_idx[~aug[test_idx]]          # test on real windows only
        if len(test_idx) == 0:
            continue
        for name, make in MODELS.items():
            if make is None:
                pred = X[test_idx, 3]                # 4th feature = last reading
            else:
                model = make().fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
            errors[name].extend(np.abs(pred - y[test_idx]))

    print("\nAverage error when forecasting the next glucose reading (mg/dL, lower is better):")
    results = {}
    for name, errs in errors.items():
        results[name] = float(np.mean(errs))
        print(f"  {name:28s} {results[name]:.1f}")

    # save the best trained model (excluding the naive one) for Mode 2
    best = min((n for n in MODELS if MODELS[n] is not None), key=lambda n: results[n])
    final = MODELS[best]().fit(X, y)
    MODELS_SAVED.mkdir(parents=True, exist_ok=True)
    with open(MODELS_SAVED / "mimic_glucose_forecaster.pkl", "wb") as f:
        pickle.dump(final, f)
    print(f"\nSaved best trained model: {best}")

    init_db()
    version_id = register_model_version(
        model_name="mimic_glucose_forecaster", n_original_rows=n_real, n_app_rows=0,
        val_auc=0.0, val_f1=0.0, val_mae=results[best],
        notes=f"{best}, mean abs error {results[best]:.1f} mg/dL, LOPO-CV")
    promote_model(version_id, "mimic_glucose_forecaster")
    print(f"Registered as mimic_glucose_forecaster, version {version_id}")



if __name__ == "__main__":
    main()
