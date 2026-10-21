import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.api.main import app

client = TestClient(app)
MODEL_TRAINED = settings.REGRESSOR_PATH.exists() and settings.CLASSIFIER_PATH.exists()


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "not a medical device" in r.json()["disclaimer"].lower()


def test_list_patients_returns_real_patients():
    r = client.get("/patients")
    assert r.status_code == 200
    patients = r.json()
    assert len(patients) > 0
    assert "patient_id" in patients[0]


def test_get_twin_for_valid_patient():
    r = client.get("/patients/VT001/twin")
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "VT001"
    assert "dynamic" in body and "glucose_mgdl" in body["dynamic"]


def test_get_twin_for_unknown_patient_returns_404():
    r = client.get("/patients/NOT_A_PATIENT/twin")
    assert r.status_code == 404


def test_history_endpoint_returns_list():
    r = client.get("/patients/VT001/history?limit=10")
    assert r.status_code == 200
    assert len(r.json()) <= 10


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_prediction_endpoint_returns_real_computation():
    r = client.get("/patients/VT001/prediction")
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body and "risk" in body
    assert body["risk"]["risk_category"] in ("LOW", "MODERATE", "HIGH")


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_simulate_endpoint_rejects_unknown_variable():
    r = client.post("/patients/VT001/simulate", json={"overrides": {"fake_variable": 1}})
    assert r.status_code == 400


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_simulate_endpoint_returns_baseline_and_simulated():
    r = client.post("/patients/VT001/simulate", json={"overrides": {"sleep_adjustment_minutes": 30}})
    assert r.status_code == 200
    body = r.json()
    assert "baseline" in body and "simulated" in body
    assert "disclaimer" in body


def test_replay_meta_returns_valid_window():
    r = client.get("/patients/VT001/replay-meta")
    assert r.status_code == 200
    body = r.json()
    assert body["total_frames"] > 0
    assert body["start_timestamp"] < body["end_timestamp"]


def test_replay_meta_unknown_patient_404():
    r = client.get("/patients/NOT_A_PATIENT/replay-meta")
    assert r.status_code == 404


def test_replay_frame_shape():
    r = client.get("/patients/VT001/replay/0")
    assert r.status_code == 200
    frame = r.json()
    assert "timestamp" in frame and "dynamic" in frame
    assert "glucose_mgdl" in frame["dynamic"]


def test_replay_frame_out_of_range_404():
    r = client.get("/patients/VT001/replay/999999")
    assert r.status_code == 404


def test_replay_frames_bulk_matches_meta_count():
    meta = client.get("/patients/VT001/replay-meta").json()
    frames = client.get("/patients/VT001/replay-frames").json()
    assert len(frames) == meta["total_frames"]
    assert frames[0]["timestamp"] == meta["start_timestamp"]
    assert frames[-1]["timestamp"] == meta["end_timestamp"]


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_replay_frames_show_risk_category_variety():
    """
    The whole point of replay is showing risk change over a day — assert
    the window actually contains more than one risk category, not just a
    constant value (would indicate the window/model isn't doing anything).
    """
    frames = client.get("/patients/VT001/replay-frames").json()
    categories = {f["risk"]["risk_category"] for f in frames if f["risk"]}
    assert len(categories) >= 2


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_recommendations_endpoint():
    r = client.get("/patients/VT001/recommendations")
    assert r.status_code == 200
    recs = r.json()
    assert isinstance(recs, list) and len(recs) >= 1
    assert "recommendation" in recs[0]
