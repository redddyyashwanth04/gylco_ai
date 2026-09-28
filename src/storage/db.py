"""
Local persistence layer for the demo app.

WHY THIS EXISTS
    NHANES and MIMIC-IV (in data/processed/) are fixed, historical training
    data -- read many times, never written to. That's a solved problem with
    plain CSV files.

    This module solves a DIFFERENT problem: when a clinician uses Mode 1 on
    a real patient today, then comes back in three months and checks the
    same patient again, the app needs to remember them. Without this, every
    check-in is disconnected from the last one, and Mode 2 could only ever
    show pre-loaded MIMIC-IV example cases -- never a real patient someone
    is actually tracking through repeated use of the app.

WHAT IT IS / ISN'T
    This is a single-file SQLite database -- no server, no separate install,
    just Python's built-in sqlite3 module. Appropriate for a prototype's
    scale (one clinician's local use, not a multi-user production system).
    It is NOT a replacement for data/processed/ -- those stay as they are.

SCHEMA
    patients: one row per named patient the app has ever seen
    visits:   one row per Mode-1 check-in, tied to a patient_id, so a
              patient with 2+ visits has a real, accumulated trajectory
              Mode 2 can plot and forecast on -- built from actual use,
              not from MIMIC-IV.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ROOT

DB_PATH = ROOT / "data" / "app_state.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    TEXT NOT NULL,
    visit_date    TEXT NOT NULL,
    hba1c         REAL,
    glucose       REAL,
    bmi           REAL,
    systolic_bp   REAL,
    diastolic_bp  REAL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS model_registry (
    version_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name        TEXT NOT NULL,        -- e.g. "nhanes_stacking_ensemble"
    trained_at        TEXT NOT NULL,
    n_original_rows   INTEGER NOT NULL,      -- how many NHANES/MIMIC rows this version trained on
    n_app_rows        INTEGER NOT NULL,      -- how many real app-collected visits were included
    val_auc            REAL,
    val_f1              REAL,
    is_active          INTEGER NOT NULL DEFAULT 0,   -- exactly one active version per model_name
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS prediction_log (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at     TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    version_id    INTEGER NOT NULL,
    patient_id    TEXT,
    prediction    TEXT NOT NULL,     -- JSON string: {"nephropathy": 0.64, ...}
    FOREIGN KEY (version_id) REFERENCES model_registry(version_id)
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def normalize_row(row, columns=None):
    """SQLite can yield either sqlite3.Row objects or legacy tuple rows. The app expects dict-like access."""
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    if columns is not None:
        return {key: value for key, value in zip(columns, row)}
    return row


def init_db():
    """Creates the database file and tables if they don't exist yet. Safe to call every app startup."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def add_visit(patient_id, hba1c=None, glucose=None, bmi=None, systolic_bp=None, diastolic_bp=None):
    """
    Records one Mode-1 check-in for a patient. Creates the patient record
    automatically on their first-ever visit.
    """
    conn = get_connection()
    now = datetime.now().isoformat()

    conn.execute(
        "INSERT OR IGNORE INTO patients (patient_id, created_at) VALUES (?, ?)",
        (patient_id, now),
    )
    conn.execute(
        """INSERT INTO visits (patient_id, visit_date, hba1c, glucose, bmi, systolic_bp, diastolic_bp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, now, hba1c, glucose, bmi, systolic_bp, diastolic_bp),
    )
    conn.commit()
    conn.close()


def get_patient_history(patient_id):
    """
    Returns every logged visit for one patient, oldest first -- this is
    what Mode 2 plots when the clinician picks a real (app-tracked) patient
    instead of a pre-loaded MIMIC-IV example case.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT visit_date, hba1c, glucose, bmi, systolic_bp, diastolic_bp "
        "FROM visits WHERE patient_id = ? ORDER BY visit_date ASC",
        (patient_id,),
    ).fetchall()
    conn.close()
    return rows


def list_patients_with_history(min_visits=2):
    """
    Returns patient_ids that have enough logged visits to plot a real trend
    -- these are the real, app-tracked patients Mode 2 can offer alongside
    the pre-loaded MIMIC-IV examples.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT patient_id, COUNT(*) as n_visits FROM visits
           GROUP BY patient_id HAVING n_visits >= ?""",
        (min_visits,),
    ).fetchall()
    conn.close()
    return rows


def register_model_version(model_name, n_original_rows, n_app_rows, val_auc, val_f1, notes=""):
    """
    Records a newly-trained candidate model version. Does NOT make it
    active -- that only happens via promote_model(), after the validation
    gate has confirmed it isn't worse than the current active version.
    Returns the new version_id.
    """
    conn = get_connection()
    now = datetime.now().isoformat()
    cur = conn.execute(
        """INSERT INTO model_registry
           (model_name, trained_at, n_original_rows, n_app_rows, val_auc, val_f1, is_active, notes)
           VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
        (model_name, now, n_original_rows, n_app_rows, val_auc, val_f1, notes),
    )
    conn.commit()
    version_id = cur.lastrowid
    conn.close()
    return version_id


def get_active_model_version(model_name):
    """Returns the registry row for the currently active version of a model, or None if none is active yet."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM model_registry WHERE model_name = ? AND is_active = 1", (model_name,)
    )
    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description] if cursor.description else None
    conn.close()
    return normalize_row(row, columns)


def promote_model(version_id, model_name):
    """
    Makes one version active and deactivates all others for the same
    model_name. Should only be called after the validation gate confirms
    the candidate isn't worse than the current active model -- this
    function itself does not judge quality, it only performs the swap.
    """
    conn = get_connection()
    conn.execute("UPDATE model_registry SET is_active = 0 WHERE model_name = ?", (model_name,))
    conn.execute("UPDATE model_registry SET is_active = 1 WHERE version_id = ?", (version_id,))
    conn.commit()
    conn.close()


def log_prediction(model_name, version_id, prediction_dict, patient_id=None):
    """
    Records every prediction the app serves -- what model version made it,
    when, and for whom (if a named patient). This is the audit trail a
    real clinical decision-support tool needs: if a prediction is
    questioned later, you can see exactly which model version produced it.
    """
    import json
    conn = get_connection()
    conn.execute(
        "INSERT INTO prediction_log (logged_at, model_name, version_id, patient_id, prediction) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), model_name, version_id, patient_id, json.dumps(prediction_dict)),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database ready at: {DB_PATH}")

    # quick smoke test
    add_visit("DEMO_PATIENT_1", hba1c=7.2, glucose=145, bmi=29.1, systolic_bp=132, diastolic_bp=84)
    add_visit("DEMO_PATIENT_1", hba1c=7.6, glucose=158, bmi=29.4, systolic_bp=135, diastolic_bp=85)

    history = get_patient_history("DEMO_PATIENT_1")
    print(f"\nDEMO_PATIENT_1 history ({len(history)} visits):")
    for v in history:
        print(" ", v)

    print("\nPatients with 2+ visits:", list_patients_with_history())
