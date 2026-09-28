"""
Retraining pipeline: turns stored patient check-ins into model improvement,
without ever silently deploying a worse model.

WHY IT WORKS THIS WAY (read before changing the promotion logic)
    A naive version of this ("retrain on every new check-in, always use
    the newest model") is a genuinely bad idea for a clinical tool: a
    handful of new, unrepresentative patient records could quietly drag
    accuracy down, and nobody would notice until predictions got worse.
    Real clinical ML systems never auto-deploy a retrained model without
    checking it first -- that's not extra caution for its own sake, it's
    the actual standard.

    So this pipeline has three stages that are kept strictly separate:
      1. TRAIN a new candidate model (original data + new app-collected data)
      2. VALIDATE the candidate against the same held-out set the current
         active model was evaluated on
      3. PROMOTE the candidate to active ONLY if it doesn't regress
         performance beyond a small tolerance -- otherwise it's logged
         and discarded, and the current active model keeps serving
         predictions.

WHY new app-collected data has a minimum-count gate
    Our own real patient check-ins (via Mode 1) start at zero and grow
    slowly. Retraining on, say, 3 new rows against an 8,153-row NHANES
    training set would have virtually no effect anyway, but retraining on
    a MIMIC-IV track with only 35 patients total is a different story --
    a handful of new rows there really could shift things, for better or
    worse. MIN_NEW_ROWS below is deliberately conservative.

HOW TO WIRE THIS UP ONCE src/models/ EXISTS
    train_fn and evaluate_fn are passed in rather than imported directly,
    so this pipeline can be built and tested now, before the actual model
    training code exists (Step 3). Once src/models/stacking_ensemble.py
    etc. are written, pass their train()/evaluate() functions in here --
    nothing in this file needs to change.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED
from src.storage.db import (
    get_connection, register_model_version, get_active_model_version, promote_model,
)

MIN_NEW_ROWS = 20          # don't bother retraining until at least this many new app check-ins exist
REGRESSION_TOLERANCE = 0.01  # candidate AUC can be at most this much worse and still get promoted
                              # (small tolerance, not zero -- a candidate that's a rounding error worse
                              #  but includes fresher real-world data is still often worth keeping)


def get_new_app_data_count():
    """How many real patient visits have been logged since the app started being used."""
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
    conn.close()
    return n


def run_retraining_cycle(model_name, train_fn, evaluate_fn, original_training_data,
                          held_out_validation_data):
    """
    The full pipeline, one call. Returns a dict summarizing what happened
    -- always returns something printable/loggable, never silently no-ops.

    train_fn(training_dataframe) -> a trained model object
    evaluate_fn(model, validation_dataframe) -> {"auc": float, "f1": float}
    """
    n_new = get_new_app_data_count()
    if n_new < MIN_NEW_ROWS:
        return {
            "status": "skipped",
            "reason": f"only {n_new} new app-collected rows so far, need {MIN_NEW_ROWS} before retraining",
        }

    # 1. TRAIN -- combine original data with real app-collected data
    conn = get_connection()
    app_rows = conn.execute("SELECT * FROM visits").fetchall()
    conn.close()
    # NOTE: app_rows currently has different columns than the NHANES/MIMIC schema
    # (patient_id, visit_date, hba1c, glucose, bmi, systolic_bp, diastolic_bp).
    # A real merge step belongs here once src/features/ defines the shared schema --
    # left as an explicit TODO rather than silently guessed at.
    combined_training_data = original_training_data  # TODO: append mapped app_rows once schema aligns

    candidate_model = train_fn(combined_training_data)

    # 2. VALIDATE -- same held-out set the current active model was scored on
    candidate_metrics = evaluate_fn(candidate_model, held_out_validation_data)

    active = get_active_model_version(model_name)
    version_id = register_model_version(
        model_name=model_name,
        n_original_rows=len(original_training_data),
        n_app_rows=n_new,
        val_auc=candidate_metrics["auc"],
        val_f1=candidate_metrics["f1"],
        notes=f"candidate trained {datetime.now().isoformat()}",
    )

    if active is None:
        # first model ever trained for this name -- nothing to compare against, promote automatically
        promote_model(version_id, model_name)
        return {"status": "promoted", "reason": "first model version, no prior baseline",
                "version_id": version_id, "metrics": candidate_metrics}

    # 3. GATE -- only promote if not meaningfully worse than what's currently serving predictions
    active_auc = active["val_auc"]
    if candidate_metrics["auc"] >= active_auc - REGRESSION_TOLERANCE:
        promote_model(version_id, model_name)
        return {
            "status": "promoted",
            "reason": f"candidate AUC {candidate_metrics['auc']:.3f} vs active {active_auc:.3f} -- within tolerance",
            "version_id": version_id, "metrics": candidate_metrics,
        }
    else:
        return {
            "status": "rejected",
            "reason": f"candidate AUC {candidate_metrics['auc']:.3f} vs active {active_auc:.3f} -- regression too large",
            "version_id": version_id, "metrics": candidate_metrics,
        }


if __name__ == "__main__":
    print(f"New app-collected rows so far: {get_new_app_data_count()}")
    print(f"Minimum needed before a retraining cycle runs: {MIN_NEW_ROWS}")
    print("\nThis script is ready to wire up once src/models/ (Step 3) provides")
    print("real train_fn / evaluate_fn implementations. Run via run_retraining_cycle(...).")
