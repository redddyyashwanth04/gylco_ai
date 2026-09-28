"""
Pretrain-then-finetune strategy for Track B's LSTM, used alongside (not
instead of) the classical augmentation in augmentation.py.

WHY THIS EXISTS
    Even after classical augmentation, 13-35 real patients is a small base
    to train an LSTM from a random initialization. This module implements
    the transfer-learning fix discussed in the architecture review: pretrain
    the LSTM on a larger, related dataset first, then fine-tune only on the
    real MIMIC-IV Demo trajectories.

PRETRAINING SOURCE: UCI "Diabetes 130-US Hospitals" dataset
    This is the SAME dataset that was deliberately rejected earlier as a
    PRIMARY dataset (see Progress_Report_Data_Foundation.md and the dataset
    detour discussion) -- it lacks real timestamps and has coarse lab bins,
    so it cannot demonstrate true time-interval forecasting on its own.
    It is reused here for a narrower, legitimate purpose: roughly 30% of its
    ~70,000 patients have 2+ encounters (via patient_nbr), which is enough
    repeat-visit STRUCTURE to teach the LSTM general sequence patterns
    before it ever sees the small, real, high-quality MIMIC-IV data.

    Download: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
    Place the extracted diabetic_data.csv in data/external/

TWO-STAGE TRAINING (not yet implemented -- see TODOs)
    Stage 1 (pretrain): train the LSTM on UCI patients with 2+ encounters,
        using encounter ORDER (not real time gaps, since UCI doesn't have
        them) as the sequence dimension. Target: whatever proxy label is
        available in UCI (e.g. readmission) -- this stage is about learning
        general sequence structure, not the final prediction task.
    Stage 2 (finetune): re-initialize the final layer(s) for the REAL target
        (complication onset), then continue training only on the real,
        augmented MIMIC-IV Demo trajectories from augmentation.py -- with a
        lower learning rate so the pretrained weights aren't overwritten
        too aggressively.

HONEST LIMITATION TO KEEP IN THE PAPER
    Pretraining on UCI teaches the model "what a sequence of hospital visits
    generally looks like," not diabetes-specific glycemic trajectory
    patterns (UCI's labs are coarse categorical bins). State this
    explicitly rather than implying the pretraining data is equivalent
    in quality to the fine-tuning data.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_MODEL_READY
import importlib
DATA_EXTERNAL = Path(__file__).resolve().parent.parent.parent / "data" / "external"

UCI_DIABETES_130_PATH = DATA_EXTERNAL / "diabetic_data.csv"


def load_pretrain_source():
    """
    Loads the UCI Diabetes-130 data and filters to patients with 2+
    encounters (via patient_nbr) -- the only rows useful for teaching the
    model general sequence structure.
    """
    import pandas as pd
    if not UCI_DIABETES_130_PATH.exists():
        raise FileNotFoundError(
            f"Pretraining source not found at {UCI_DIABETES_130_PATH}. "
            "Download from https://archive.ics.uci.edu/dataset/296/ and place there."
        )
    df = pd.read_csv(UCI_DIABETES_130_PATH)
    repeat_patients = df.groupby("patient_nbr").filter(lambda g: len(g) >= 2)
    print(f"Loaded {len(repeat_patients)} rows from "
          f"{repeat_patients['patient_nbr'].nunique()} repeat-encounter patients "
          f"(out of {df['patient_nbr'].nunique()} total)")
    return repeat_patients


def pretrain_lstm(pretrain_df, model_config=None):
    # TODO: build encounter sequences from UCI data (ordered by encounter_id,
    # no real timestamps available), train LSTM on a proxy target (e.g.
    # readmission), return the trained model for stage 2.
    raise NotImplementedError("Pretraining loop not yet implemented -- depends on lstm_trajectory.py")


def finetune_on_real_data(pretrained_model, real_sequences, learning_rate=1e-4):
    # TODO: swap the final layer for the real target (complication onset),
    # continue training only on real (+ augmented) MIMIC-IV Demo data at a
    # lower learning rate than pretraining used.
    raise NotImplementedError("Fine-tuning loop not yet implemented -- depends on lstm_trajectory.py")


if __name__ == "__main__":
    print("This module pretrains on UCI Diabetes-130 and fine-tunes on real MIMIC-IV Demo data.")
    print(f"Expected pretraining source location: {UCI_DIABETES_130_PATH}")
    print(f"Exists: {UCI_DIABETES_130_PATH.exists()}")
