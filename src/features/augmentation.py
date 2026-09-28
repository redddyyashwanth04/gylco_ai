"""
Augmentation strategy for Track B -- the FINAL, committed dataset strategy
for this project. Full MIMIC-IV credentialed access was applied for and
declined (PhysioNet requires a verifiable institutional reference email);
rather than keep pursuing that, the MIMIC-IV Demo cohort is the permanent
Track B data source, expanded via the techniques below.

PRIMARY SIGNAL: GLUCOSE, NOT HBA1C -- confirmed from real data, not assumed
    Checking the actual mimic_trajectories.csv data changed the plan:
    glucose is checked far more often than HbA1c in ICU care (point-of-care
    testing, sometimes multiple times a day), so it is a dramatically richer
    signal in this dataset. Real, verified counts:
        - HbA1c:   20 of 35 patients have any reading, only 7 have 2+
        - Glucose: all 35 patients have readings, 30 have 5+, one patient
                   alone has 185 real readings, 1,212 real readings total
    Track B's sequence model should be built around GLUCOSE trajectories as
    the primary input, with HbA1c layered in as a secondary, coarser
    feature where available -- not the reverse, which was the original
    (less-informed) framing before this was checked against real data.

THREE TECHNIQUES, IN ORDER OF PREFERENCE (all operate on REAL patient data --
none of these invent a synthetic patient; they create variations of real
trajectories)

    1. WINDOW SLICING
       For patients with many readings (verified: one patient has 185
       glucose readings), extract overlapping fixed-length sub-sequences
       as separate training examples, instead of just one long sequence
       per patient. Real, computed result on this data: 35 patients, 1,212
       real readings -> 375 windows (min_window=5, stride=3).

    2. JITTERING
       Add small Gaussian noise to each lab value (e.g. +/- 2%) to create
       several slightly-varied copies of each window. Keep noise small
       enough that it never crosses a clinically meaningful threshold --
       clinical plausibility must be checked, not assumed. Combined with
       window slicing above: 375 windows x 5 jitter copies = 1,875 real
       patient-derived training examples.

    3. TIME WARPING
       Slightly stretch or compress the intervals between a patient's
       readings (not the values themselves) to simulate natural variation
       in measurement timing.

WHAT NOT TO DO
    Do not train a GAN (e.g. CTGAN) on this trajectory data -- 35 real
    patients is too few to train a generative model well; it would likely
    just memorize and reproduce the training patients. GAN-based synthesis
    is more appropriate for larger, cross-sectional data (e.g. NHANES),
    not this small longitudinal set. See Base_Papers_Deep_Dive_Blueprint.md
    and the architecture discussion in reports/ for the full reasoning.

REPORTING RULE (do not violate this when writing up results)
    Every result produced using augmented data MUST report both the real
    patient count and the augmented example count separately, e.g.
    "35 real patients, 1,212 real glucose readings, expanded to 1,875
    training examples via window-slicing and jittering." Never let a table
    imply N=1,875 real patients. LOPO-CV (src/models/lopo_cv.py) must keep
    ALL windows/jitter-copies from one real patient in the same fold --
    splitting one patient's augmented examples across train and test would
    leak information and invalidate the evaluation.
"""

import numpy as np
import pandas as pd


def jitter(values, noise_std_frac=0.02, n_copies=3, seed=None):
    """
    Returns n_copies noisy variants of a 1D array of real lab values.
    noise_std_frac is the noise standard deviation as a fraction of each
    value (e.g. 0.02 = 2% noise) -- deliberately small and multiplicative,
    so a higher baseline value gets proportionally more noise than a low one,
    which is more physiologically realistic than fixed additive noise.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    copies = []
    for _ in range(n_copies):
        noise = rng.normal(0, noise_std_frac * np.abs(values))
        copies.append(values + noise)
    return copies


def window_slice(sequence, min_window=5, stride=2):
    """
    Given one patient's full ordered sequence (list/array of readings),
    returns overlapping FIXED-LENGTH sub-sequences of exactly min_window
    length. A patient with fewer than min_window readings returns just the
    original sequence (no slicing possible) -- callers should handle that
    case rather than assume every patient produces multiple windows.
    """
    n = len(sequence)
    if n < min_window:
        return [sequence]
    windows = []
    start = 0
    while start + min_window <= n:
        windows.append(sequence[start:start + min_window])
        start += stride
    return windows


def time_warp(timestamps, warp_std_frac=0.1, seed=None):
    """
    Given a patient's real visit timestamps (as day-offsets from their first
    visit), returns a warped copy where the GAPS between visits are
    stretched/compressed by a small random factor -- the ORDER of visits
    never changes, only the spacing. warp_std_frac controls how much the
    gaps can vary (0.1 = up to ~10% stretch/compress on average).
    """
    rng = np.random.default_rng(seed)
    timestamps = np.asarray(timestamps, dtype=float)
    gaps = np.diff(timestamps, prepend=timestamps[0])
    warp_factors = rng.normal(1.0, warp_std_frac, size=len(gaps))
    warp_factors = np.clip(warp_factors, 0.5, 1.5)  # never compress/stretch more than 2x
    warped_gaps = gaps * warp_factors
    return np.cumsum(warped_gaps)


def augment_patient_trajectory(patient_df, min_window=5, stride=3, n_jitter_copies=5, seed=None):
    """
    Full pipeline for one patient's trajectory. Applies window slicing
    first, then jittering to each resulting window. Returns a list of
    DataFrames, each tagged with 'source_patient' and 'is_augmented' so the
    reporting rule in this module's docstring can be honored downstream --
    every training example traces back to exactly one real patient, and it
    is always clear which examples are the real, unmodified original.

    patient_df: a DataFrame for ONE patient with columns
        ['charttime', 'valuenum'], already sorted by charttime.
    Returns: list of DataFrames, each with columns
        ['charttime', 'valuenum', 'source_patient', 'is_augmented'].
    """
    subject_id = patient_df.attrs.get("subject_id", "unknown")
    values = patient_df["valuenum"].tolist()
    times = patient_df["charttime"].tolist()

    value_windows = window_slice(values, min_window=min_window, stride=stride)
    n = len(values)
    time_windows = []
    start = 0
    while start + min_window <= n:
        time_windows.append(times[start:start + min_window])
        start += stride
    if not time_windows:  # too few readings to window -- use the whole sequence once
        time_windows = [times]

    results = []
    for w_idx, (val_window, time_window) in enumerate(zip(value_windows, time_windows)):
        # the first (unsliced or first-sliced) window, un-jittered, is the "real" example
        is_first = (w_idx == 0)
        real_df = pd.DataFrame({
            "charttime": time_window,
            "valuenum": val_window,
            "source_patient": subject_id,
            "is_augmented": False,
        })
        results.append(real_df)

        jittered_copies = jitter(val_window, n_copies=n_jitter_copies, seed=seed)
        for copy_vals in jittered_copies:
            aug_df = pd.DataFrame({
                "charttime": time_window,
                "valuenum": copy_vals,
                "source_patient": subject_id,
                "is_augmented": True,
            })
            results.append(aug_df)

    return results


def augment_full_cohort(trajectories_df, lab_name="glucose", min_window=5, stride=3, n_jitter_copies=5, seed=None):
    """
    Runs augment_patient_trajectory() across every patient in a
    mimic_trajectories.csv-style long-format DataFrame, for one lab
    (glucose by default -- see module docstring for why glucose, not HbA1c,
    is the primary signal). Returns a single concatenated DataFrame ready
    for sequence modeling, plus a summary dict with the real-vs-augmented
    counts required by the reporting rule.
    """
    lab_df = trajectories_df[trajectories_df["lab_name"] == lab_name].copy()
    lab_df["charttime"] = pd.to_datetime(lab_df["charttime"])

    all_results = []
    for subject_id, group in lab_df.groupby("subject_id"):
        group = group.sort_values("charttime")
        group.attrs["subject_id"] = subject_id
        all_results.extend(
            augment_patient_trajectory(group, min_window=min_window, stride=stride,
                                        n_jitter_copies=n_jitter_copies, seed=seed)
        )

    combined = pd.concat(all_results, ignore_index=True)
    n_real_windows = combined[~combined["is_augmented"]].shape[0] // min_window
    n_total_examples = len(combined) // min_window
    summary = {
        "real_patients": lab_df["subject_id"].nunique(),
        "real_readings": len(lab_df),
        "total_windows": n_real_windows,
        "total_training_examples": n_total_examples,
    }
    return combined, summary


if __name__ == "__main__":
    # smoke test with synthetic example values (NOT real patient data --
    # just here to confirm the basic functions run)
    example_glucose = [140, 155, 148, 162, 170, 165, 158, 145, 150]
    jittered = jitter(example_glucose, n_copies=2, seed=42)
    print("Original:", example_glucose)
    print("Jittered copy 1:", np.round(jittered[0], 1))

    windows = window_slice(example_glucose, min_window=5, stride=2)
    print(f"\n{len(windows)} fixed-length windows from a 9-reading sequence:")
    for w in windows:
        print(" ", w)

    # full pipeline test against REAL data, if available
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    try:
        from config import MIMIC_TRAJECTORIES
        if MIMIC_TRAJECTORIES.exists():
            print(f"\n--- Running full pipeline on real data: {MIMIC_TRAJECTORIES} ---")
            traj = pd.read_csv(MIMIC_TRAJECTORIES)
            combined, summary = augment_full_cohort(traj, lab_name="glucose", seed=42)
            print(f"Real patients:            {summary['real_patients']}")
            print(f"Real glucose readings:    {summary['real_readings']}")
            print(f"Windows (real, unjittered): {summary['total_windows']}")
            print(f"Total training examples (real + jittered): {summary['total_training_examples']}")
    except ImportError:
        pass
