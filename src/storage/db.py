"""
Local persistence layer for the demo app (SQLite).

Tables: patients, visits (now with a per-visit risk snapshot), model_registry,
prediction_log. Existing databases are migrated automatically by init_db().
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import ROOT

DB_PATH = ROOT / "data" / "app_state.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id        TEXT NOT NULL,
    visit_date        TEXT NOT NULL,
    hba1c             REAL,
    glucose           REAL,
    bmi               REAL,
    systolic_bp       REAL,
    diastolic_bp      REAL,
    risks_json        TEXT,
    model_version_id  INTEGER,
    note              TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS model_registry (
    version_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name        TEXT NOT NULL,
    trained_at        TEXT NOT NULL,
    n_original_rows   INTEGER NOT NULL,
    n_app_rows        INTEGER NOT NULL,
    val_auc            REAL,
    val_f1              REAL,
    val_mae             REAL,
    is_active          INTEGER NOT NULL DEFAULT 0,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS prediction_log (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at     TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    version_id    INTEGER NOT NULL,
    patient_id    TEXT,
    prediction    TEXT NOT NULL,
    FOREIGN KEY (version_id) REFERENCES model_registry(version_id)
);
"""


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def normalize_row(row, columns=None):
    """SQLite can yield sqlite3.Row objects or legacy tuples. The app expects dict-like access."""
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    if columns is not None:
        return {key: value for key, value in zip(columns, row)}
    return row


def _migrate(conn):
    # visits table — columns added over time
    visit_cols = {r[1] for r in conn.execute("PRAGMA table_info(visits)")}
    if "risks_json" not in visit_cols:
        conn.execute("ALTER TABLE visits ADD COLUMN risks_json TEXT")
    if "model_version_id" not in visit_cols:
        conn.execute("ALTER TABLE visits ADD COLUMN model_version_id INTEGER")
    if "note" not in visit_cols:
        conn.execute("ALTER TABLE visits ADD COLUMN note TEXT")

    # model_registry table — val_mae added in migration v2
    registry_cols = {r[1] for r in conn.execute("PRAGMA table_info(model_registry)")}
    if "val_mae" not in registry_cols:
        conn.execute("ALTER TABLE model_registry ADD COLUMN val_mae REAL")


def init_db():
    """Creates the database and tables if missing, and migrates older files. Safe to call on every startup."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    conn.close()


def add_visit(patient_id, hba1c=None, glucose=None, bmi=None, systolic_bp=None,
              diastolic_bp=None, risks=None, model_version_id=None, note=None):
    """Records one check-in. Stores the model's risk snapshot and optional clinician note."""
    conn = get_connection()
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO patients (patient_id, created_at) VALUES (?, ?)",
        (patient_id, now),
    )
    conn.execute(
        """INSERT INTO visits (patient_id, visit_date, hba1c, glucose, bmi, systolic_bp,
                               diastolic_bp, risks_json, model_version_id, note)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, now, hba1c, glucose, bmi, systolic_bp, diastolic_bp,
         json.dumps(risks) if risks else None, model_version_id, note or None),
    )
    conn.commit()
    conn.close()


def get_patient_history(patient_id):
    """Every logged visit for one patient, oldest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT visit_date, hba1c, glucose, bmi, systolic_bp, diastolic_bp, risks_json "
        "FROM visits WHERE patient_id = ? ORDER BY visit_date ASC, visit_id ASC",
        (patient_id,),
    ).fetchall()
    conn.close()
    return rows


def list_patients_with_history(min_visits=2):
    conn = get_connection()
    rows = conn.execute(
        """SELECT patient_id, COUNT(*) as n_visits FROM visits
           GROUP BY patient_id HAVING n_visits >= ?""",
        (min_visits,),
    ).fetchall()
    conn.close()
    return rows


def list_patient_summaries():
    conn = get_connection()
    rows = conn.execute(
        """SELECT patient_id, COUNT(*) AS n_visits, MIN(visit_date) AS first_visit,
                  MAX(visit_date) AS last_visit
           FROM visits GROUP BY patient_id ORDER BY last_visit DESC"""
    ).fetchall()
    conn.close()
    return rows


def count_predictions():
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
    conn.close()
    return n


def register_model_version(model_name, n_original_rows, n_app_rows,
                           val_auc, val_f1, val_mae=None, notes=""):
    """Records a candidate model version. Does NOT make it active (see promote_model).

    val_mae: optional numeric MAE (used by Track B forecasters). Stored in a
             dedicated column so api_server.py can read it without regex-parsing notes.
    """
    conn = get_connection()
    now = datetime.now().isoformat()
    cur = conn.execute(
        """INSERT INTO model_registry
           (model_name, trained_at, n_original_rows, n_app_rows,
            val_auc, val_f1, val_mae, is_active, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
        (model_name, now, n_original_rows, n_app_rows,
         val_auc, val_f1, val_mae, notes),
    )
    conn.commit()
    version_id = cur.lastrowid
    conn.close()
    return version_id


def get_active_model_version(model_name):
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM model_registry WHERE model_name = ? AND is_active = 1", (model_name,)
    )
    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description] if cursor.description else None
    conn.close()
    return normalize_row(row, columns)


def promote_model(version_id, model_name):
    conn = get_connection()
    conn.execute("UPDATE model_registry SET is_active = 0 WHERE model_name = ?", (model_name,))
    conn.execute("UPDATE model_registry SET is_active = 1 WHERE version_id = ?", (version_id,))
    conn.commit()
    conn.close()


def log_prediction(model_name, version_id, prediction_dict, patient_id=None):
    """Audit trail: which model version made which prediction, when, for whom."""
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