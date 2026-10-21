import copy

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.state import DigitalTwinEngine
from src.models.predict import ModelNotTrainedError
from src.simulation.whatif import apply_overrides, run_simulation

MODEL_TRAINED = True
try:
    from src.config import settings
    MODEL_TRAINED = settings.REGRESSOR_PATH.exists() and settings.CLASSIFIER_PATH.exists()
except Exception:
    MODEL_TRAINED = False


def _history_df(n_steps=60):
    ts = pd.date_range("2026-08-01T00:00:00", periods=n_steps, freq="5min")
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "patient_id": "VT001", "timestamp": ts,
        "glucose_mgdl": 140 + np.cumsum(rng.normal(0, 2, n_steps)),
        "heart_rate_bpm": 72 + rng.normal(0, 1, n_steps),
        "hrv_ms": 48 + rng.normal(0, 1, n_steps),
        "steps": rng.integers(0, 50, n_steps),
        "activity_state": "light", "is_asleep": False,
        "age": 55, "sex": "M", "bmi": 31.0,
        "diagnosis": "T2D", "diabetes_duration_years": 5, "hba1c_pct": 8.0, "medication": "metformin",
    })


def test_apply_overrides_does_not_mutate_input_dict():
    features = {"steps_roll_sum_60m": 100.0, "activity_intensity_roll_mean_60m": 0.5,
                "sleep_minutes_prev_night": 300.0, "glucose_mgdl": 150.0,
                "glucose_roll_mean_15m": 148.0, "glucose_roll_mean_30m": 147.0}
    original = copy.deepcopy(features)
    apply_overrides(features, {"activity_boost_steps_60m": 500})
    assert features == original  # input untouched


def test_apply_overrides_activity_boost_increases_steps_and_intensity():
    features = {"steps_roll_sum_60m": 100.0, "activity_intensity_roll_mean_60m": 0.5}
    sim = apply_overrides(features, {"activity_boost_steps_60m": 1000})
    assert sim["steps_roll_sum_60m"] == 1100.0
    assert sim["activity_intensity_roll_mean_60m"] > 0.5


def test_apply_overrides_sleep_adjustment():
    features = {"sleep_minutes_prev_night": 300.0}
    sim = apply_overrides(features, {"sleep_adjustment_minutes": 60})
    assert sim["sleep_minutes_prev_night"] == 360.0
    sim2 = apply_overrides(features, {"sleep_adjustment_minutes": -400})
    assert sim2["sleep_minutes_prev_night"] == 0.0  # clamped, never negative


def test_apply_overrides_unknown_variable_rejected_by_run_simulation():
    features = {"glucose_mgdl": 150.0}
    with pytest.raises(ValueError, match="Unsupported simulation variable"):
        run_simulation(features, {"not_a_real_variable": 5})


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_run_simulation_never_mutates_real_twin_state():
    """
    This is the critical safety property required by the spec:
    baseline = twin.snapshot(); result = simulator.run(...); assert twin.snapshot() == baseline
    """
    df = _history_df()
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    baseline_snapshot = engine.snapshot()

    fv = engine.current_state().features_vector()
    run_simulation(fv, {"activity_boost_steps_60m": 2000, "sleep_adjustment_minutes": 90})

    assert engine.snapshot() == baseline_snapshot


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_run_simulation_produces_different_result_for_different_overrides():
    df = _history_df()
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    fv = engine.current_state().features_vector()

    result_small = run_simulation(fv, {"activity_boost_steps_60m": 100})
    result_large = run_simulation(fv, {"activity_boost_steps_60m": 3000})

    assert (
        result_small["simulated"]["prediction"]["predicted_glucose_mgdl"]
        != result_large["simulated"]["prediction"]["predicted_glucose_mgdl"]
    )


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_run_simulation_includes_disclaimer():
    df = _history_df()
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    fv = engine.current_state().features_vector()
    result = run_simulation(fv, {"sleep_adjustment_minutes": 30})
    assert "not a clinical treatment recommendation" in result["disclaimer"].lower()
