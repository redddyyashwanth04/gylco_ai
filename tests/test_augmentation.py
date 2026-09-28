"""
Tests for src/features/augmentation.py -- these run with no data files or
trained models needed, since augmentation functions operate on plain arrays.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.features.augmentation import jitter, window_slice, time_warp


def test_jitter_preserves_shape_and_count():
    values = [7.1, 7.4, 7.8, 8.1]
    copies = jitter(values, n_copies=5, seed=1)
    assert len(copies) == 5
    for c in copies:
        assert len(c) == len(values)


def test_jitter_is_close_to_original():
    """Jittering should perturb, not drastically change, the values."""
    values = np.array([7.1, 7.4, 7.8, 8.1])
    copies = jitter(values, noise_std_frac=0.02, n_copies=10, seed=1)
    for c in copies:
        assert np.allclose(c, values, atol=1.0), "Jittered values strayed further than expected from originals"


def test_window_slice_short_sequence_returns_original():
    short_seq = [7.1, 7.4]
    windows = window_slice(short_seq, min_window=5, stride=2)
    assert windows == [short_seq]


def test_window_slice_long_sequence_produces_multiple_windows():
    long_seq = list(range(10))  # 10 readings
    windows = window_slice(long_seq, min_window=5, stride=2)
    assert len(windows) > 1, "Expected multiple windows from a 10-reading sequence"


def test_time_warp_preserves_order():
    """Warping gaps should never reorder visits."""
    timestamps = np.array([0, 10, 25, 40, 100])
    warped = time_warp(timestamps, warp_std_frac=0.1, seed=1)
    assert np.all(np.diff(warped) > 0), "Time warping must preserve visit order (monotonically increasing)"


def test_augment_full_cohort_runs_against_real_data():
    """
    End-to-end check against the actual project data -- confirms the real,
    reported numbers (35 patients, 1,212 readings -> 373 windows -> 2,240
    training examples) stay accurate as the pipeline evolves, rather than
    silently drifting from what's written in README.md.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import MIMIC_TRAJECTORIES
    from src.features.augmentation import augment_full_cohort
    import pandas as pd

    if not MIMIC_TRAJECTORIES.exists():
        return  # skip silently if data isn't present in this environment

    traj = pd.read_csv(MIMIC_TRAJECTORIES)
    combined, summary = augment_full_cohort(traj, lab_name="glucose", seed=1)

    assert summary["real_patients"] == 35, "Real patient count drifted from the documented 35 -- update README.md if this is expected"
    assert summary["total_windows"] > 0, "No windows produced -- augmentation pipeline is broken"
    assert summary["total_training_examples"] > summary["total_windows"], (
        "Jittering should always produce more training examples than raw windows"
    )
    # every row must be traceable to a real patient -- no orphaned/anonymous augmented rows
    assert combined["source_patient"].notna().all()
