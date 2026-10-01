"""
HTTP integration tests for the four primary API endpoints:
  POST /api/predict    — complication risk from patient values
  POST /api/explain    — SHAP drivers for the same values
  POST /api/forecast   — next glucose reading from a sequence
  POST /api/visits     — save a visit to the local patient database

The server is started in a background thread at module import and torn down
after all tests finish. Tests use real model artifacts from models_saved/
so they also catch serialisation and feature-column drift.

Run: pytest tests/test_api.py -v
"""

import json
import socket
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── helpers ────────────────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:5099"   # port 5099 avoids colliding with the real server on 5001


def _free_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read())


# ── server fixture ─────────────────────────────────────────────────────────────

_server_thread = None
_server_obj = None


def _start_test_server():
    """Boot api_server on port 5099 in a daemon thread."""
    from http.server import ThreadingHTTPServer
    from src.api_server import ApiHandler
    from src.storage.db import init_db

    init_db()
    srv = ThreadingHTTPServer(("127.0.0.1", 5099), ApiHandler)
    global _server_obj
    _server_obj = srv
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    global _server_thread
    _server_thread = t
    # Wait until the port actually accepts connections
    for _ in range(30):
        if not _free_port_available(5099):
            break
        time.sleep(0.1)


@pytest.fixture(scope="session", autouse=True)
def api_server():
    """Session-scoped fixture: start the server once, yield, then shut it down."""
    if not _free_port_available(5099):
        pytest.skip("Port 5099 is already in use — cannot start test server.")
    _start_test_server()
    yield
    if _server_obj:
        _server_obj.shutdown()


# ── shared payload ─────────────────────────────────────────────────────────────

EXAMPLE_VALUES = {
    "RIDAGEYR": 58,
    "LBXGH": 7.4,
    "LBXGLU": 150,
    "BMXBMI": 31.0,
    "BPXOSY1": 142,
    "BPXODI1": 88,
    "LBDLDL": 130,
    "LBDHDD": 42,
}

# ── /api/health ────────────────────────────────────────────────────────────────

def test_health_returns_connected():
    result = _get("/api/health")
    assert result.get("connected") is True, f"Health check not connected: {result}"
    assert "modelName" in result
    assert "featureCount" in result
    assert isinstance(result["featureCount"], int)
    assert result["featureCount"] > 0


# ── POST /api/predict ──────────────────────────────────────────────────────────

def test_predict_returns_three_risks():
    result = _post("/api/predict", {"values": EXAMPLE_VALUES})
    assert "risks" in result, f"No risks key: {result}"
    risks = result["risks"]
    for target in ("hypertension", "nephropathy", "cardiovascular"):
        assert target in risks, f"Missing target {target}"
        assert 0.0 <= risks[target] <= 1.0, f"{target} risk out of [0, 1]: {risks[target]}"


def test_predict_includes_model_name():
    result = _post("/api/predict", {"values": EXAMPLE_VALUES})
    assert "modelName" in result
    assert isinstance(result["modelName"], str)
    assert len(result["modelName"]) > 0


def test_predict_includes_recommendations():
    result = _post("/api/predict", {"values": EXAMPLE_VALUES})
    assert "recommendations" in result
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) >= 1


def test_predict_with_optional_fields():
    """Optional fields (sex, smoking, sedentary, income) must not cause errors."""
    values = {**EXAMPLE_VALUES, "RIAGENDR": 1, "SMQ020": 2, "PAD680": 9.0, "INDFMPIR": 1.8}
    result = _post("/api/predict", {"values": values})
    assert "risks" in result


def test_predict_rejects_missing_required_field():
    """Age is required — omitting it should return 400."""
    bad = {k: v for k, v in EXAMPLE_VALUES.items() if k != "RIDAGEYR"}
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post("/api/predict", {"values": bad})
    assert exc_info.value.code == 400


def test_predict_rejects_out_of_range_value():
    """Age 200 is out of range — should return 400."""
    bad = {**EXAMPLE_VALUES, "RIDAGEYR": 200}
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post("/api/predict", {"values": bad})
    assert exc_info.value.code == 400


def test_predict_trend_returned_for_known_patient():
    """After saving a visit, the next predict call should return trends."""
    patient_id = "TEST-TREND-001"
    # First, save a visit so there is prior history
    _post("/api/visits", {"patientId": patient_id, "values": EXAMPLE_VALUES})
    # Now predict — trends should be in the response
    result = _post("/api/predict", {"patientId": patient_id, "values": EXAMPLE_VALUES})
    assert "trends" in result
    # trends may be None if only one visit exists; just confirm the key is present


# ── POST /api/explain ──────────────────────────────────────────────────────────

def test_explain_returns_factors_for_all_targets():
    result = _post("/api/explain", {"values": EXAMPLE_VALUES})
    assert "risks" in result
    assert "factors" in result
    for target in ("hypertension", "nephropathy", "cardiovascular"):
        assert target in result["factors"], f"No factors for {target}"
        factors = result["factors"][target]
        assert isinstance(factors, list), f"Factors for {target} is not a list"
        # Each factor must have the keys the frontend uses
        if factors:
            for key in ("feature", "label", "shap", "value", "entered"):
                assert key in factors[0], f"Factor missing key '{key}'"


def test_explain_returns_narrative():
    result = _post("/api/explain", {"values": EXAMPLE_VALUES})
    assert "narrative" in result
    for target in ("hypertension", "nephropathy", "cardiovascular"):
        assert target in result["narrative"]
        assert isinstance(result["narrative"][target], str)
        assert len(result["narrative"][target]) > 0


def test_explain_optional_fields_marked_not_entered():
    """Fields not provided should appear as entered=False in the SHAP output."""
    minimal = {"RIDAGEYR": 45, "LBXGH": 6.0, "LBXGLU": 110}
    result = _post("/api/explain", {"values": minimal})
    # At least some factors should be marked as not entered (median-filled)
    all_factors = [
        f for target_factors in result["factors"].values() for f in target_factors
    ]
    not_entered = [f for f in all_factors if not f.get("entered")]
    assert len(not_entered) > 0, "Expected some factors to be median-filled (entered=False)"


# ── POST /api/forecast ─────────────────────────────────────────────────────────

GLUCOSE_READINGS = [126.0, 139.0, 145.0, 158.0, 149.0, 171.0]


def test_forecast_returns_value():
    result = _post("/api/forecast", {"readings": GLUCOSE_READINGS})
    assert "forecast" in result, f"No forecast key: {result}"
    assert isinstance(result["forecast"], float)
    # Physiologically plausible range
    assert 50 <= result["forecast"] <= 500, f"Forecast out of range: {result['forecast']}"


def test_forecast_returns_last_observed_and_readings_used():
    result = _post("/api/forecast", {"readings": GLUCOSE_READINGS})
    assert "lastObserved" in result
    assert result["lastObserved"] == GLUCOSE_READINGS[-1]
    assert "readingsUsed" in result
    assert result["readingsUsed"] >= 4


def test_forecast_returns_expected_error():
    result = _post("/api/forecast", {"readings": GLUCOSE_READINGS})
    # expectedError may be None if not yet registered, but the key must exist
    assert "expectedError" in result


def test_forecast_rejects_too_few_readings():
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post("/api/forecast", {"readings": [120.0, 140.0]})
    assert exc_info.value.code == 400


def test_forecast_rejects_non_list():
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post("/api/forecast", {"readings": "not-a-list"})
    assert exc_info.value.code == 400


# ── POST /api/visits ───────────────────────────────────────────────────────────

def test_save_visit_returns_saved_true():
    result = _post("/api/visits", {
        "patientId": "TEST-SAVE-001",
        "values": EXAMPLE_VALUES,
    })
    assert result.get("saved") is True
    assert result.get("patientId") == "TEST-SAVE-001"


def test_save_visit_persists_and_readable():
    patient_id = "TEST-SAVE-002"
    _post("/api/visits", {"patientId": patient_id, "values": EXAMPLE_VALUES})
    visits = _get("/api/visits")
    patient_visits = [v for v in visits["visits"] if v["patient_id"] == patient_id]
    assert len(patient_visits) >= 1


def test_save_visit_persists_note():
    patient_id = "TEST-NOTE-001"
    note_text = "Patient reports improved diet this month."
    _post("/api/visits", {
        "patientId": patient_id,
        "values": EXAMPLE_VALUES,
        "note": note_text,
    })
    visits = _get("/api/visits")
    patient_visits = [v for v in visits["visits"] if v["patient_id"] == patient_id]
    assert len(patient_visits) >= 1
    assert patient_visits[0]["note"] == note_text


def test_save_visit_rejects_missing_patient_id():
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post("/api/visits", {"patientId": "", "values": EXAMPLE_VALUES})
    assert exc_info.value.code == 400


def test_save_visit_rejects_patient_id_too_long():
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post("/api/visits", {"patientId": "X" * 41, "values": EXAMPLE_VALUES})
    assert exc_info.value.code == 400


def test_save_visit_rejects_note_too_long():
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post("/api/visits", {
            "patientId": "TEST-LONGNOTE",
            "values": EXAMPLE_VALUES,
            "note": "A" * 2001,
        })
    assert exc_info.value.code == 400
