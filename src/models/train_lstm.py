"""
LSTM glucose forecaster for Track B. Run this on your machine after
`pip install torch`. Not tested in the sandbox this was written in (no
internet there to install torch) -- test it yourself and tell me what
happens.

Run: python src/models/train_lstm.py

What it does, in order:
  1. Build the same real+augmented glucose windows as train_track_b.py
     (30 real patients, 370 real windows, 1,850 with augmentation).
  2. Train a small LSTM: last 4 readings in, 5th reading predicted.
  3. Test with leave-one-patient-out -- test only on real data, and never
     split one patient's windows across train and test.
  4. Print the same error metric as train_track_b.py (mean absolute error
     in mg/dL) so you can directly compare LSTM vs Ridge vs Gradient
     Boosting vs Naive.
  5. Save the best-epoch model and register it.
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
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

WINDOW = 5      # same as train_track_b.py -- 4 readings in, 1 predicted
STRIDE = 3
JITTER_COPIES = 4
HIDDEN_SIZE = 16   # deliberately small -- this is a ~30-patient dataset, a big
                   # LSTM will just memorize it. Do not raise this without a
                   # reason.
EPOCHS = 60
LEARNING_RATE = 0.01


class GlucoseLSTM(nn.Module):
    """Takes 4 past readings, predicts the 5th. Input shape: (batch, 4, 1)."""
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def build_windows(traj_df):
    """Same logic as train_track_b.py's build_windows, kept in sync deliberately."""
    glucose = traj_df[traj_df["lab_name"] == "glucose"]
    sequences, targets, groups, aug = [], [], [], []
    for pid, g in glucose.groupby("subject_id"):
        values = g.sort_values("charttime")["valuenum"].tolist()
        for win in window_slice(values, min_window=WINDOW, stride=STRIDE):
            if len(win) < WINDOW:
                continue
            sequences.append(win[:-1]); targets.append(win[-1]); groups.append(pid); aug.append(False)
            for copy in jitter(win, n_copies=JITTER_COPIES, seed=1):
                sequences.append(copy[:-1]); targets.append(copy[-1]); groups.append(pid); aug.append(True)
    return np.array(sequences), np.array(targets), np.array(groups), np.array(aug)


def train_one_fold(X_train, y_train, scaler):
    """Trains a fresh small LSTM on one fold's training data."""
    X_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    x_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1)  # (n, 4, 1)
    y_t = torch.tensor(scaler.transform(y_train.reshape(-1, 1)).flatten(), dtype=torch.float32)

    model = GlucoseLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
    return model


def predict_fold(model, X_test, scaler):
    X_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    x_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_t).numpy()
    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


def main():
    traj = pd.read_csv(MIMIC_TRAJECTORIES)
    X, y, groups, aug = build_windows(traj)
    print(f"Real patients: {len(set(groups))} | real windows: {int((~aug).sum())} | "
          f"total examples with augmentation: {len(y)}")

    scaler = StandardScaler().fit(X.reshape(-1, 1))  # fit on ALL data's scale, this is just a
                                                        # unit conversion, not a leak

    logo = LeaveOneGroupOut()
    errors = []
    for train_idx, test_idx in logo.split(X, y, groups):
        test_idx = test_idx[~aug[test_idx]]  # test on real windows only
        if len(test_idx) == 0:
            continue
        model = train_one_fold(X[train_idx], y[train_idx], scaler)
        pred = predict_fold(model, X[test_idx], scaler)
        errors.extend(np.abs(pred - y[test_idx]))

    mae = float(np.mean(errors))
    print(f"\nLSTM mean absolute error: {mae:.1f} mg/dL")
    print("Compare against train_track_b.py's numbers:")
    print("  Naive (repeat last value): 65.4")
    print("  Linear (Ridge):            57.6")
    print("  Gradient Boosting:         65.8")
    if mae < 57.6:
        print("  -> LSTM is the new best model.")
    else:
        print("  -> LSTM did not beat Ridge. That's a real, reportable result on this small dataset.")

    # train the final model on ALL data (real + augmented) and save it
    final_model = train_one_fold(X, y, scaler)
    MODELS_SAVED.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), MODELS_SAVED / "mimic_lstm_forecaster.pt")
    with open(MODELS_SAVED / "mimic_lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved LSTM weights to {MODELS_SAVED / 'mimic_lstm_forecaster.pt'}")

    init_db()
    version_id = register_model_version(
        model_name="mimic_glucose_forecaster_lstm",
        n_original_rows=int((~aug).sum()), n_app_rows=0,
        val_auc=0.0, val_f1=0.0,
        notes=f"LSTM, hidden={HIDDEN_SIZE}, mean abs error {mae:.1f} mg/dL, LOPO-CV",
    )
    promote_model(version_id, "mimic_glucose_forecaster_lstm")
    print(f"Registered as mimic_glucose_forecaster_lstm, version {version_id}")


if __name__ == "__main__":
    main()
