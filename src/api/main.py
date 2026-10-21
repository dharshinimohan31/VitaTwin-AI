"""
VitaTwin AI — FastAPI backend (Phase 8).

Every endpoint calls the real underlying modules (preprocessing, digital
twin, models, risk, explainability, simulation) — there is no hardcoded
JSON anywhere here. Twins are built on first request per patient and
cached in-process for the life of the server (fine for a PoC; a real
deployment would back this with the twin_states table instead).

Run: uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.digital_twin.recommendations import recommend
from src.digital_twin.risk import assess
from src.digital_twin.state import DigitalTwinEngine
from src.explainability.explain import explain
from src.features.engineering import FEATURE_COLUMNS, add_features
from src.models.predict import ModelNotTrainedError, predict
from src.preprocessing.pipeline import run_pipeline
from src.simulation.whatif import SIMULATABLE_VARIABLES, run_simulation

# Live Twin replay (Phase 10): how many 5-minute steps the replay window
# covers. 288 steps = 24 hours at the dataset's 5-min resolution.
REPLAY_WINDOW_STEPS = 288

app = FastAPI(
    title="VitaTwin AI",
    description="Predict. Simulate. Prevent. — research/educational prototype, not a medical device.",
    version="0.1.0",
)

# Allow the standalone web dashboard (src/dashboard/web/, served separately
# from this API, e.g. on a different port via `python -m http.server`) to
# call these endpoints from the browser. Open CORS is acceptable here only
# because this is a local-only research PoC with no auth/sensitive data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulateRequest(BaseModel):
    overrides: dict[str, float] = Field(..., description="Keys must be in SIMULATABLE_VARIABLES")


@lru_cache(maxsize=1)
def _fused_data() -> pd.DataFrame:
    fused, _ = run_pipeline()
    return fused


@lru_cache(maxsize=32)
def _get_engine(patient_id: str) -> DigitalTwinEngine:
    fused = _fused_data()
    if patient_id not in set(fused["patient_id"]):
        raise KeyError(patient_id)
    return DigitalTwinEngine.create_from_history(patient_id, fused)


def _require_engine(patient_id: str) -> DigitalTwinEngine:
    try:
        return _get_engine(patient_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown patient_id '{patient_id}'")


def _jsafe(value):
    """Same numpy->plain-Python coercion as digital_twin.state._safe, kept local
    to avoid importing a private helper across modules."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


@lru_cache(maxsize=32)
def _featured_patient_df(patient_id: str) -> pd.DataFrame:
    """
    Feature-engineer one patient's FULL timeline once. Every feature here is
    strictly backward-looking (rolling windows, expanding baseline — see
    tests/test_features.py's no-leakage test), so row i's features are
    identical whether computed incrementally (as DigitalTwinEngine does) or
    over the whole series at once, as here. Precomputing once makes replay
    playback fast — each frame then costs one cheap model inference, not a
    full feature recompute.
    """
    fused = _fused_data()
    patient_df = fused[fused["patient_id"] == patient_id].sort_values("timestamp")
    if patient_df.empty:
        raise KeyError(patient_id)
    return add_features(patient_df).reset_index(drop=True)


@lru_cache(maxsize=32)
def _replay_frames(patient_id: str) -> tuple[dict, ...]:
    featured = _featured_patient_df(patient_id)
    window_start = max(0, len(featured) - REPLAY_WINDOW_STEPS)
    window = featured.iloc[window_start:].reset_index(drop=True)

    frames = []
    for _, row in window.iterrows():
        fv = {k: _jsafe(row.get(k)) for k in FEATURE_COLUMNS}
        frame = {
            "timestamp": str(row["timestamp"]),
            "dynamic": {
                "glucose_mgdl": _jsafe(row.get("glucose_mgdl")),
                "heart_rate_bpm": _jsafe(row.get("heart_rate_bpm")),
                "hrv_ms": _jsafe(row.get("hrv_ms")),
                "steps": _jsafe(row.get("steps")),
                "activity_state": row.get("activity_state"),
                "is_asleep": bool(row.get("is_asleep", False)),
            },
            "derived": {
                "baseline_glucose_deviation_pct": _jsafe(row.get("baseline_glucose_deviation_pct")),
                "sleep_minutes_prev_night": _jsafe(row.get("sleep_minutes_prev_night")),
            },
            "prediction": None,
            "risk": None,
        }
        try:
            pred = predict(fv)
            frame["prediction"] = pred
            frame["risk"] = assess(fv, pred["excursion_probability"])
        except ModelNotTrainedError:
            pass  # replay still works for vitals; prediction/risk stay null until trained
        frames.append(frame)
    return tuple(frames)


@app.get("/health")
def health():
    return {"status": "ok", "service": "VitaTwin AI", "disclaimer": "Research/educational prototype — not a medical device."}


@app.get("/patients")
def list_patients():
    fused = _fused_data()
    rows = fused.drop_duplicates("patient_id")[["patient_id", "age", "sex", "bmi", "hba1c_pct", "diagnosis"]]
    return rows.to_dict("records")


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    engine = _require_engine(patient_id)
    snap = engine.snapshot()
    return {"patient_id": patient_id, "demographics": snap["demographics"], "historical": snap["historical"]}


@app.get("/patients/{patient_id}/twin")
def get_twin(patient_id: str):
    engine = _require_engine(patient_id)
    return engine.snapshot()


@app.get("/patients/{patient_id}/history")
def get_history(patient_id: str, limit: int = 50):
    engine = _require_engine(patient_id)
    return engine.history(limit=limit)


@app.get("/patients/{patient_id}/prediction")
def get_prediction(patient_id: str):
    engine = _require_engine(patient_id)
    fv = engine.current_state().features_vector()
    try:
        pred = predict(fv)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    risk = assess(fv, pred["excursion_probability"])
    return {"prediction": pred, "risk": risk}


@app.get("/patients/{patient_id}/risk")
def get_risk(patient_id: str):
    return get_prediction(patient_id)["risk"]


@app.get("/patients/{patient_id}/explain")
def get_explanation(patient_id: str):
    engine = _require_engine(patient_id)
    fv = engine.current_state().features_vector()
    try:
        return explain(fv, FEATURE_COLUMNS)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Explainability unavailable: {e}")


@app.get("/patients/{patient_id}/recommendations")
def get_recommendations(patient_id: str):
    engine = _require_engine(patient_id)
    fv = engine.current_state().features_vector()
    try:
        pred = predict(fv)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    risk = assess(fv, pred["excursion_probability"])
    return recommend(fv, risk)


@app.get("/patients/{patient_id}/replay-meta")
def get_replay_meta(patient_id: str):
    """Metadata for the Live Twin replay window — how many frames, covering what time span."""
    _require_engine(patient_id)
    try:
        frames = _replay_frames(patient_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown patient_id '{patient_id}'")
    return {
        "total_frames": len(frames),
        "start_timestamp": frames[0]["timestamp"],
        "end_timestamp": frames[-1]["timestamp"],
        "interval_minutes": settings.READING_INTERVAL_MINUTES,
        "model_trained": frames[0]["prediction"] is not None,
    }


@app.get("/patients/{patient_id}/replay-frames")
def get_replay_frames(patient_id: str):
    """
    All frames in the replay window at once, for the dashboard to prefetch
    and animate client-side (smooth playback with no per-frame network
    round-trip). Same precomputed, real data as the per-step endpoint.
    """
    _require_engine(patient_id)
    try:
        frames = _replay_frames(patient_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown patient_id '{patient_id}'")
    return list(frames)


@app.get("/patients/{patient_id}/replay/{step}")
def get_replay_frame(patient_id: str, step: int):
    """
    One frame of the Live Twin replay: real vitals + real prediction/risk at
    that historical moment, precomputed from the same trained model as every
    other endpoint — not synthesized for animation.
    """
    _require_engine(patient_id)
    try:
        frames = _replay_frames(patient_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown patient_id '{patient_id}'")
    if step < 0 or step >= len(frames):
        raise HTTPException(status_code=404, detail=f"step out of range 0..{len(frames) - 1}")
    return frames[step]


@app.get("/patients/{patient_id}/simulate-options")
def get_simulate_options(patient_id: str):
    _require_engine(patient_id)
    return SIMULATABLE_VARIABLES


@app.post("/patients/{patient_id}/simulate")
def post_simulate(patient_id: str, body: SimulateRequest):
    engine = _require_engine(patient_id)
    fv = engine.current_state().features_vector()
    try:
        return run_simulation(fv, body.overrides)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
