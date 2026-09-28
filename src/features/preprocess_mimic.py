"""
Preprocessing for MIMIC-IV (Track B) -- turns the augmented trajectory
data (from augmentation.py) into sequences a PyTorch LSTM/GRU can consume.

PRIMARY SIGNAL: GLUCOSE, NOT HBA1C -- confirmed from real data, see
augmentation.py's module docstring for the full reasoning. Glucose has
readings for all 35 diabetic patients (1,212 total, one patient has 185);
HbA1c only has readings for 20 patients (7 with 2+). Build sequences around
glucose as the primary channel; layer in HbA1c as a secondary, coarser
feature where available, not as the primary sequence.

Input:  augmented glucose data from augmentation.augment_full_cohort()
        (373 real windows -> 2,240 training examples, computed and verified
        against the real project data -- see README.md's Track B section)
Output: data/model_ready/mimic_sequences.pt (or .npz -- pick one, document it)

STEPS (not yet implemented -- see TODOs)
    1. Call augmentation.augment_full_cohort(traj_df, lab_name="glucose")
       to get the combined real+augmented dataset and its summary counts.
    2. For each window (already fixed-length at min_window=5 from
       augmentation.py), compute time_since_window_start from charttime.
    3. Track which rows are real vs. augmented (the 'is_augmented' column)
       and which patient each window came from ('source_patient') -- both
       are required for the LOPO-CV fold-assignment rule (see lopo_cv.py
       and the reporting rule in augmentation.py's docstring): all windows
       from one real patient, real and augmented, MUST stay in the same
       fold.
    4. Join with the complication flags from mimic_patient_summary.csv as
       the prediction target (nephropathy and cardiovascular only --
       neuropathy and retinopathy are too sparse in this cohort per the
       earlier prevalence check in Progress_Report_Data_Foundation.md).
    5. Convert to padded tensors (all windows are already fixed-length, so
       padding is simpler here than it would be for variable-length
       sequences -- no truncation decision needed since window_slice()
       already fixed the length).
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import MIMIC_TRAJECTORIES, MIMIC_PATIENT_SUMMARY, DATA_MODEL_READY
from src.features.augmentation import augment_full_cohort

WINDOW_LENGTH = 5  # matches augmentation.py's default min_window


def load_trajectories():
    return pd.read_csv(MIMIC_TRAJECTORIES)


def build_sequences(traj_df, lab_name="glucose"):
    combined, summary = augment_full_cohort(traj_df, lab_name=lab_name)
    print(f"Real patients: {summary['real_patients']} | "
          f"Real readings: {summary['real_readings']} | "
          f"Windows: {summary['total_windows']} | "
          f"Training examples: {summary['total_training_examples']}")
    # TODO: convert `combined` (long format: charttime, valuenum,
    # source_patient, is_augmented) into padded (batch, WINDOW_LENGTH, 1)
    # tensors, join complication targets from mimic_patient_summary.csv,
    # and return everything lopo_cv.py + lstm_trajectory.py need.
    raise NotImplementedError("Tensor conversion not yet implemented -- augmentation step above is done")


def main():
    traj = load_trajectories()
    n_patients = traj["subject_id"].nunique()
    print(f"Loaded {len(traj)} trajectory rows across {n_patients} patients")
    try:
        build_sequences(traj)
    except NotImplementedError as e:
        print(f"(Stopped at: {e})")


if __name__ == "__main__":
    main()
