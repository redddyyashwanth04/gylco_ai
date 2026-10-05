"""Local HTTP adapter for the standalone clinician frontend.

Run from the project root with:
    python -m src.api_server

Binds to loopback only; not suitable for network or production use.
"""

import json
import pickle
import sys
from math import isfinite
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import MODELS_SAVED
from src.models.predict import (
    TARGETS,
    compute_risk_delta,
    generate_clinical_recommendations,
    load_active_model,
    predict_patient,
)
from src.models.train_lstm import predict_from_checkpoint
from src.storage.db import (
    add_visit,
    count_predictions,
    get_active_model_version,
    get_connection,
    get_patient_history,
    init_db,
    list_patient_summaries,
    log_prediction,
)

HOST = "127.0.0.1"
PORT = 5001
ALLOWED_ORIGINS = {
    "http://127.0.0.1:5001",
    "http://localhost:5001",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
}
FRONTEND_DIR = ROOT / "frontend"
MAX_BODY_BYTES = 64 * 1024
VALUE_RANGES = {
    "RIDAGEYR": (18, 100),
    "LBXGH": (3, 15),
    "LBXGLU": (50, 400),
    "BMXBMI": (10, 70),
    "BPXOSY1": (70, 250),
    "BPXODI1": (40, 150),
    "LBDLDL": (30, 300),
    "LBDHDD": (15, 120),
    "RIAGENDR": (1, 2),
    "SMQ020": (1, 3),
    "PAD680": (0, 24),
    "INDFMPIR": (0, 5),
}


def validate_values(values, required_fields=()):
    """Validates numeric clinical input. Empty optional fields (None) are dropped."""
    if not isinstance(values, dict):
        raise ValueError("Patient values are required.")
    for field in required_fields:
        if values.get(field) is None:
            raise ValueError(f"{field} is required for the model input.")
    normalized = dict(values)
    for field, value in list(normalized.items()):
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a numeric clinical value.")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be numeric.") from error
        if not isfinite(number):
            raise ValueError(f"{field} must be finite.")
        if field in VALUE_RANGES:
            minimum, maximum = VALUE_RANGES[field]
            if not minimum <= number <= maximum:
                raise ValueError(f"{field} must be between {minimum} and {maximum}.")
        normalized[field] = number
    return {k: v for k, v in normalized.items() if v is not None}


def forecaster_expected_error(model_name: str) -> float | None:
    """Return the LOPO-CV MAE for the active version of a forecaster.

    Reads the dedicated val_mae column added in db migration v2.
    Falls back to regex-parsing the notes field for older registry rows
    that pre-date the column (keeps the server working across upgrades).
    """
    active = get_active_model_version(model_name)
    if not active:
        return None
    # Primary: dedicated numeric column (added in db migration v2)
    val_mae = active.get("val_mae")
    if val_mae is not None:
        return float(val_mae)
    # Fallback: parse free-text notes for rows written before migration v2
    import re
    notes = active.get("notes") or ""
    match = re.search(r"(?:mean abs error|MAE)\s+([\d.]+)", notes)
    return float(match.group(1)) if match else None



class ApiHandler(BaseHTTPRequestHandler):
    server_version = "CarepathLocalAPI/1.1"

    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_cors_headers(self):
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def reject_untrusted_origin(self):
        origin = self.headers.get("Origin")
        # Same-origin requests (browser navigation, no Origin header) are always allowed.
        if origin and origin not in ALLOWED_ORIGINS:
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "Origin is not allowed."})
            return True
        return False

    _MIME = {
        ".html": "text/html; charset=utf-8",
        ".js":   "application/javascript; charset=utf-8",
        ".css":  "text/css; charset=utf-8",
        ".ico":  "image/x-icon",
        ".png":  "image/png",
        ".svg":  "image/svg+xml",
        ".json": "application/json; charset=utf-8",
    }

    def serve_static(self, url_path):
        """Serve files from the frontend/ directory for browser requests."""
        # Strip leading slash; map bare "/" or "/index.html" to index.html
        rel = url_path.lstrip("/") or "index.html"
        # Route /login.html explicitly
        file_path = (FRONTEND_DIR / rel).resolve()
        # Prevent path traversal outside frontend/
        try:
            file_path.relative_to(FRONTEND_DIR.resolve())
        except ValueError:
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden."})
            return
        if not file_path.is_file():
            # For any non-file, serve index.html (SPA fallback)
            file_path = FRONTEND_DIR / "index.html"
        ext = file_path.suffix.lower()
        mime = self._MIME.get(ext, "application/octet-stream")
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise ValueError("Send request data as application/json.")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Request body is missing or too large.")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_OPTIONS(self):
        if self.reject_untrusted_origin():
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.reject_untrusted_origin():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self.get_health()
            elif path == "/api/visits":
                self.get_visits()
            elif path == "/api/models":
                self.get_models()
            elif path == "/api/patients":
                self.get_patients()
            elif path.startswith("/api/patients/") and path.endswith("/visits"):
                patient_id = unquote(path[len("/api/patients/"):-len("/visits")].strip("/"))
                self.get_patient_visits(patient_id)
            elif path.startswith("/api/"):
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            else:
                # Serve the frontend SPA (index.html, login.html, styles.css, app.js, …)
                self.serve_static(path)
        except (ValueError, TypeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception:
            self.log_error("GET request failed for %s", path)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "The local service could not complete this request."})

    def do_POST(self):
        if self.reject_untrusted_origin():
            return
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
            if path == "/api/predict":
                self.post_prediction(payload)
            elif path == "/api/explain":
                self.post_explain(payload)
            elif path == "/api/visits":
                self.post_visit(payload)
            elif path == "/api/forecast":
                self.post_forecast(payload)
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception:
            self.log_error("POST request failed for %s", path)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "The local service could not complete this request."})

    # ---------- GET handlers ----------

    def get_health(self):
        try:
            model_name, _, feature_columns = load_active_model()
            active = get_active_model_version("nhanes_active")
            lstm_available  = (MODELS_SAVED / "mimic_lstm_forecaster.pt").exists()
            ridge_available = (MODELS_SAVED / "mimic_glucose_forecaster.pkl").exists()
            forecast_available = lstm_available or ridge_available
            forecast_model = "LSTM v4" if lstm_available else ("Ridge" if ridge_available else None)
            self.send_json(HTTPStatus.OK, {
                "connected": True,
                "modelName": model_name,
                "featureCount": len(feature_columns),
                "registryVersion": active["version_id"] if active else None,
                "forecastAvailable": forecast_available,
                "forecastModel": forecast_model,
            })
        except Exception:
            self.log_error("Model health check failed")
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "connected": False,
                "error": "The saved NHANES model could not be loaded.",
            })

    def get_visits(self):
        connection = get_connection()
        try:
            rows = connection.execute(
                """SELECT patient_id, visit_date, hba1c, glucose, bmi,
                          systolic_bp, diastolic_bp, risks_json, note
                   FROM visits ORDER BY visit_date DESC, visit_id DESC"""
            ).fetchall()
            self.send_json(HTTPStatus.OK, {"visits": [dict(row) for row in rows]})
        finally:
            connection.close()

    def get_patient_visits(self, patient_id):
        if not patient_id:
            raise ValueError("Patient ID is required.")
        rows = get_patient_history(patient_id)
        self.send_json(HTTPStatus.OK, {
            "patientId": patient_id,
            "visits": [dict(row) for row in rows],
        })

    def get_patients(self):
        self.send_json(HTTPStatus.OK, {
            "patients": [dict(r) for r in list_patient_summaries()],
            "predictionCount": count_predictions(),
        })

    def get_models(self):
        connection = get_connection()
        try:
            rows = connection.execute(
                """SELECT model_name, version_id, trained_at, n_original_rows,
                          n_app_rows, val_auc, val_f1, val_mae, is_active, notes
                   FROM model_registry ORDER BY model_name, trained_at DESC"""
            ).fetchall()
            self.send_json(HTTPStatus.OK, {"models": [dict(row) for row in rows]})
        finally:
            connection.close()

    def get_cases(self):
        """Kept for backward compatibility — returns empty list. Cases removed from UI."""
        self.send_json(HTTPStatus.OK, {"cases": []})

    # ---------- POST handlers ----------

    def post_prediction(self, payload):
        values = validate_values(payload.get("values"), ("RIDAGEYR", "LBXGH", "LBXGLU"))

        risks, model_name, _ = predict_patient(values)
        recommendations = generate_clinical_recommendations(values, risks)

        # Trend = change versus the risks stored at the patient's previous visit.
        trends = None
        patient_id = str(payload.get("patientId", "")).strip()
        if patient_id:
            history = get_patient_history(patient_id)
            if history and history[-1]["risks_json"]:
                previous = json.loads(history[-1]["risks_json"])
                trends = {}
                for target in TARGETS:
                    delta, direction = compute_risk_delta({target: risks[target]}, previous)
                    trends[target] = {"delta": delta, "direction": direction}

        active = get_active_model_version("nhanes_active")
        if active:
            log_prediction("nhanes_active", active["version_id"], risks, patient_id or None)

        self.send_json(HTTPStatus.OK, {
            "risks": risks,
            "modelName": model_name,
            "recommendations": recommendations,
            "trends": trends,
        })

    def post_explain(self, payload):
        values = validate_values(payload.get("values"), ("RIDAGEYR", "LBXGH", "LBXGLU"))
        from src.explainability.explain import explain_patient
        self.send_json(HTTPStatus.OK, explain_patient(values))

    def post_visit(self, payload):
        patient_id = str(payload.get("patientId", "")).strip()
        if not patient_id:
            raise ValueError("A patient ID is required to save a visit to the patient record.")
        if len(patient_id) > 40:
            raise ValueError("Patient ID cannot exceed 40 characters.")
        values = validate_values(payload.get("values"))
        note = str(payload.get("note", "")).strip() or None
        if note and len(note) > 2000:
            raise ValueError("Visit note cannot exceed 2000 characters.")

        # Risk snapshot is computed server-side; the client's numbers are not trusted.
        risks, version_id = None, None
        try:
            risks, _, _ = predict_patient(values)
            active = get_active_model_version("nhanes_active")
            version_id = active["version_id"] if active else None
        except Exception:
            self.log_error("Risk snapshot failed while saving visit")

        add_visit(
            patient_id,
            hba1c=values.get("LBXGH"),
            glucose=values.get("LBXGLU"),
            bmi=values.get("BMXBMI"),
            systolic_bp=values.get("BPXOSY1"),
            diastolic_bp=values.get("BPXODI1"),
            risks=risks,
            model_version_id=version_id,
            note=note,
        )
        self.send_json(HTTPStatus.CREATED, {"saved": True, "patientId": patient_id})

    def post_forecast(self, payload):
        readings = payload.get("readings")
        if not isinstance(readings, list):
            raise ValueError("A list of glucose readings is required.")

        MIN_READINGS = 4
        if len(readings) < MIN_READINGS:
            raise ValueError(f"At least {MIN_READINGS} glucose readings are required.")

        readings = [float(v) for v in readings]
        gap_hours = payload.get("gap_hours")
        if gap_hours is not None:
            if not isinstance(gap_hours, list):
                gap_hours = None
            else:
                gap_hours = [float(g) for g in gap_hours]

        lstm_path  = MODELS_SAVED / "mimic_lstm_forecaster.pt"
        ridge_path = MODELS_SAVED / "mimic_glucose_forecaster.pkl"

        forecast: float | None = None
        forecaster_name = "unavailable"
        expected_error: float | None = None

        if lstm_path.exists():
            try:
                forecast = predict_from_checkpoint(lstm_path, readings, gap_hours)
                forecaster_name = "LSTM v4 + Ridge blend"
                expected_error = forecaster_expected_error("mimic_glucose_forecaster_lstm")
            except Exception as exc:
                self.log_error("LSTM forecast failed (%s); falling back to Ridge", exc)

        if forecast is None and ridge_path.exists():
            from src.models.train_track_b import WINDOW as RIDGE_WINDOW, make_features
            with open(ridge_path, "rb") as fh:
                ridge = pickle.load(fh)
            forecast = float(ridge.predict([make_features(readings[-(RIDGE_WINDOW - 1):])])[0])
            forecaster_name = "Ridge"
            expected_error = forecaster_expected_error("mimic_glucose_forecaster")

        if forecast is None:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": "No forecast model artifact is available. "
                         "Run src/models/train_lstm.py first.",
            })
            return

        self.send_json(HTTPStatus.OK, {
            "forecast": forecast,
            "lastObserved": readings[-1],
            "readingsUsed": min(len(readings), 4),
            "expectedError": expected_error,
            "forecasterName": forecaster_name,
        })


def main():
    init_db()
    server = ThreadingHTTPServer((HOST, PORT), ApiHandler)
    print(f"Carepath local API + UI listening at http://{HOST}:{PORT}")
    print(f"  Open http://{HOST}:{PORT}/ in your browser to start.")
    print("Loopback-only prototype service. Do not expose this server to a network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Carepath local API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()