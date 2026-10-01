"""
Synthetic MIMIC-IV glucose trajectory generator.

Generates N synthetic "patients" whose glucose dynamics are statistically
anchored to the 30 real MIMIC-IV Demo patients.  The synthetic data is
used for pre-training Track B; the model is then fine-tuned (or evaluated)
exclusively on real patient windows so evaluation is never contaminated.

HOW IT WORKS
------------
1. Fit a profile for each real patient:
     - baseline mean, std, min, max
     - AR(1) autocorrelation (momentum)
     - linear trend slope (rising / falling / flat)
     - time-gap distribution (median gap between consecutive readings,
       and the spread around it)

2. Bootstrap-sample 1000 profiles from those 30 with replacement.
   Each sampled profile is slightly perturbed (within observed ranges)
   so the synthetic patients are not identical copies.

3. Simulate a glucose trajectory for each synthetic patient using a
   mean-reverting AR(1) process anchored to that patient's profile:
       g[t] = clamp(
           mean + rho*(g[t-1] - mean) + noise,
           GLUCOSE_MIN, GLUCOSE_MAX
       )
   where noise ~ N(0, sigma*(1-rho^2)^0.5) -- matches the real residual std.

4. Sample realistic time gaps from a log-normal fitted to the real gap
   distribution (heavy-tailed: most gaps are short, a few are very long).

5. Save to data/augmented/synthetic_mimic_trajectories.csv in the same
   schema as data/processed/mimic_trajectories.csv so train_lstm.py and
   train_all.py can consume it directly.

REPORTING RULE (non-negotiable)
--------------------------------
Every result using this data MUST report:
    n_real_patients=30,  n_synthetic_patients=N
Never let a table imply N synthetic patients are real.

Run:
    python src/data_prep/generate_synthetic_mimic.py
    python src/data_prep/generate_synthetic_mimic.py --n 500   # fewer patients
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_AUGMENTED, MIMIC_TRAJECTORIES

# -- physiological bounds -----------------------------------------------------
GLUCOSE_MIN = 40.0    # mg/dL  (severe hypoglycaemia floor)
GLUCOSE_MAX = 500.0   # mg/dL  (extreme hyperglycaemia ceiling)
MIN_READINGS = 20     # minimum readings per synthetic patient
MAX_READINGS = 120    # maximum readings per synthetic patient


# =============================================================================
# Step 1 -- Fit real patient profiles
# =============================================================================

def fit_profiles(traj_df: pd.DataFrame) -> list[dict]:
    """Extract a statistical profile from each real patient with 5+ readings."""
    g = (
        traj_df[traj_df["lab_name"] == "glucose"]
        .dropna(subset=["valuenum"])
        .copy()
    )
    g["charttime"] = pd.to_datetime(g["charttime"])

    profiles = []
    for pid, grp in g.groupby("subject_id"):
        grp = grp.sort_values("charttime")
        vals = grp["valuenum"].to_numpy(float)
        if len(vals) < 5:
            continue

        # time gaps in hours
        gaps = (
            grp["charttime"].diff().dt.total_seconds().dropna().to_numpy() / 3600
        )
        gaps = gaps[gaps > 0]  # drop zero-gaps (same-minute duplicates)

        # AR(1) autocorrelation -- momentum of the series
        s = pd.Series(vals)
        rho = float(s.autocorr(1)) if len(vals) > 3 else 0.5
        rho = np.clip(rho, -0.95, 0.95)   # keep it numerically stable

        # linear trend (mg/dL per reading)
        trend = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])

        profiles.append({
            "subject_id": pid,
            "n_readings":  len(vals),
            "mean":        float(vals.mean()),
            "std":         float(max(vals.std(), 10.0)),   # floor 10 mg/dL
            "min_val":     float(vals.min()),
            "max_val":     float(vals.max()),
            "rho":         rho,
            "trend":       trend,
            # gap distribution: fit log-normal
            "gap_log_mean": float(np.log1p(gaps).mean()) if len(gaps) else 2.0,
            "gap_log_std":  float(np.log1p(gaps).std())  if len(gaps) > 1 else 0.5,
        })

    return profiles


# =============================================================================
# Step 2 -- Bootstrap + perturb profiles
# =============================================================================

def sample_profiles(
    profiles: list[dict],
    n_synthetic: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Bootstrap-sample n_synthetic profiles, adding small perturbations."""
    base = rng.choice(profiles, size=n_synthetic, replace=True)
    sampled = []
    for p in base:
        p = dict(p)   # don't mutate original
        # Perturb the baseline mean within ±20% of the observed population std
        pop_std = np.std([q["mean"] for q in profiles])
        p["mean"]    += rng.normal(0, pop_std * 0.2)
        p["mean"]     = float(np.clip(p["mean"], 80, 350))

        # Perturb std within ±25%
        p["std"]      = float(np.clip(p["std"] * rng.uniform(0.75, 1.25), 10, 150))

        # Perturb rho slightly
        p["rho"]      = float(np.clip(p["rho"] + rng.normal(0, 0.05), -0.9, 0.9))

        # Perturb trend slightly (most synthetic patients should be roughly flat)
        p["trend"]    = float(rng.normal(0, 2.0))

        # Vary trajectory length
        p["n_readings"] = int(rng.integers(MIN_READINGS, MAX_READINGS + 1))

        sampled.append(p)
    return sampled


# =============================================================================
# Step 3+4 -- Simulate trajectory
# =============================================================================

def simulate_trajectory(
    profile: dict,
    patient_id: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate a glucose trajectory using a mean-reverting AR(1) process."""
    n   = profile["n_readings"]
    mu  = profile["mean"]
    sig = profile["std"]
    rho = profile["rho"]

    # Residual noise std: sigma_eps = sig * sqrt(1 - rho^2)
    eps_std = sig * np.sqrt(max(1 - rho ** 2, 0.01))

    # Starting glucose near the patient's mean (±0.5 std)
    g = float(np.clip(mu + rng.normal(0, sig * 0.5), GLUCOSE_MIN, GLUCOSE_MAX))

    values = []
    for i in range(n):
        values.append(g)
        trend_nudge = profile["trend"] * 0.1   # dampen trend so it doesn't explode
        noise = rng.normal(0, eps_std)
        g_next = mu + rho * (g - mu) + trend_nudge + noise
        g = float(np.clip(g_next, GLUCOSE_MIN, GLUCOSE_MAX))

    # Generate timestamps: start at a synthetic admission time, then add
    # log-normal gaps (same distribution as real patients)
    log_gap_mean = profile["gap_log_mean"]
    log_gap_std  = max(profile["gap_log_std"], 0.1)
    raw_gaps = rng.normal(log_gap_mean, log_gap_std, size=n)
    gap_hours = np.expm1(np.clip(raw_gaps, 0, 7))   # back to hours, max ~1096 h
    gap_hours = np.clip(gap_hours, 0.25, 72)          # 15 min to 3 days

    start = pd.Timestamp("2100-01-01") + pd.to_timedelta(
        int(rng.integers(0, 365)), unit="D"
    )
    timestamps = [start]
    for gh in gap_hours[1:]:
        timestamps.append(timestamps[-1] + pd.to_timedelta(gh, unit="h"))

    return pd.DataFrame({
        "subject_id": patient_id,
        "charttime":  timestamps,
        "lab_name":   "glucose",
        "valuenum":   values,
    })


# =============================================================================
# Main
# =============================================================================

def generate(n_synthetic: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate n_synthetic patients and return a DataFrame in mimic schema."""
    if not MIMIC_TRAJECTORIES.exists():
        raise FileNotFoundError(
            f"{MIMIC_TRAJECTORIES} not found. "
            "Run src/data_prep/build_mimic_trajectories.py first."
        )

    traj_df  = pd.read_csv(MIMIC_TRAJECTORIES)
    profiles = fit_profiles(traj_df)
    print(f"Real patient profiles fitted: {len(profiles)}")

    rng = np.random.default_rng(seed)
    synthetic_profiles = sample_profiles(profiles, n_synthetic, rng)

    # Synthetic patient IDs start at 99001 to avoid colliding with real IDs
    SYN_ID_OFFSET = 99001
    frames = []
    for i, profile in enumerate(synthetic_profiles):
        pid = SYN_ID_OFFSET + i
        df  = simulate_trajectory(profile, pid, rng)
        frames.append(df)

    synthetic_df = pd.concat(frames, ignore_index=True)
    synthetic_df["is_synthetic"] = True

    # --- validation checks ---
    total_readings = len(synthetic_df)
    per_patient    = synthetic_df.groupby("subject_id")["valuenum"]
    print(f"Synthetic patients     : {n_synthetic}")
    print(f"Total synthetic readings: {total_readings}")
    print(f"Readings per patient   : min={per_patient.count().min()}  "
          f"max={per_patient.count().max()}  "
          f"mean={per_patient.count().mean():.0f}")
    print(f"Glucose range          : {synthetic_df['valuenum'].min():.0f} -- "
          f"{synthetic_df['valuenum'].max():.0f} mg/dL")
    print(f"Glucose mean           : {synthetic_df['valuenum'].mean():.1f}  "
          f"std={synthetic_df['valuenum'].std():.1f}")

    return synthetic_df


def save(synthetic_df: pd.DataFrame) -> Path:
    """Save to data/augmented/synthetic_mimic_trajectories.csv."""
    DATA_AUGMENTED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_AUGMENTED / "synthetic_mimic_trajectories.csv"
    synthetic_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print("REPORTING NOTE: n_real_patients=30, n_synthetic_patients="
          f"{synthetic_df['subject_id'].nunique()}")
    print("Never report synthetic patients as real in any table or paper.")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MIMIC glucose trajectories")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of synthetic patients (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Generating {args.n} synthetic patients (seed={args.seed})")
    print("=" * 60)
    df = generate(n_synthetic=args.n, seed=args.seed)
    save(df)


if __name__ == "__main__":
    main()
