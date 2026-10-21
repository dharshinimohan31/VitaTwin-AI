import json

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.state import DigitalTwinEngine
from src.features.engineering import FEATURE_COLUMNS


def _history_df(n_steps=60):
    ts = pd.date_range("2026-08-01T00:00:00", periods=n_steps, freq="5min")
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "patient_id": "VT001",
        "timestamp": ts,
        "glucose_mgdl": 120 + np.cumsum(rng.normal(0, 2, n_steps)),
        "heart_rate_bpm": 70 + rng.normal(0, 1, n_steps),
        "hrv_ms": 50 + rng.normal(0, 1, n_steps),
        "steps": rng.integers(0, 50, n_steps),
        "activity_state": "light",
        "is_asleep": False,
        "age": 50, "sex": "F", "bmi": 28.0,
        "diagnosis": "T2D", "diabetes_duration_years": 2, "hba1c_pct": 7.0, "medication": "metformin",
    })


def test_create_from_history_sets_current_state_to_latest_row():
    df = _history_df()
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    snap = engine.snapshot()
    assert snap["timestamp"] == str(df["timestamp"].iloc[-1])
    assert snap["dynamic"]["glucose_mgdl"] == pytest.approx(df["glucose_mgdl"].iloc[-1])


def test_create_from_history_raises_for_unknown_patient():
    df = _history_df()
    with pytest.raises(ValueError):
        DigitalTwinEngine.create_from_history("NOT_A_PATIENT", df)


def test_snapshot_is_json_serializable():
    df = _history_df()
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    json.dumps(engine.snapshot())  # must not raise


def test_history_grows_with_each_row_and_is_ordered():
    df = _history_df()
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    hist = engine.history()
    assert len(hist) == len(df)
    timestamps = [h["timestamp"] for h in hist]
    assert timestamps == sorted(timestamps)


def test_twin_state_changes_when_new_data_supplied():
    df = _history_df(n_steps=40)
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    before = engine.snapshot()["dynamic"]["glucose_mgdl"]

    new_row = df.iloc[[-1]].copy()
    new_row["timestamp"] = new_row["timestamp"] + pd.Timedelta(minutes=5)
    new_row["glucose_mgdl"] = before + 50.0  # clearly different
    extended = pd.concat([df, new_row], ignore_index=True)

    engine.update(new_row.iloc[0].to_dict(), extended)
    after = engine.snapshot()["dynamic"]["glucose_mgdl"]
    assert after != before
    assert after == pytest.approx(before + 50.0)


def test_features_vector_has_no_missing_keys_after_create():
    df = _history_df(n_steps=60)
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    fv = engine.current_state().features_vector()
    missing = [k for k in FEATURE_COLUMNS if k not in fv or fv[k] is None]
    assert missing == [], f"features_vector missing: {missing}"


def test_fork_does_not_mutate_original_state():
    df = _history_df()
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    original_glucose = engine.current_state().dynamic["glucose_mgdl"]

    forked = engine.fork()
    forked.current_state().dynamic["glucose_mgdl"] = -999.0
    forked.current_state().derived["glucose_roll_mean_15m"] = -999.0

    assert engine.current_state().dynamic["glucose_mgdl"] == original_glucose
    assert engine.current_state().dynamic["glucose_mgdl"] != -999.0


def test_fork_history_is_independent():
    df = _history_df()
    engine = DigitalTwinEngine.create_from_history("VT001", df)
    original_len = len(engine.history())
    forked = engine.fork()
    forked._history.append(forked.current_state())
    assert len(engine.history()) == original_len
