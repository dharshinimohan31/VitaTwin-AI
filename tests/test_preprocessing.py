import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import build_fused_timeline, clean_sensor_stream


def _base_rows():
    return [
        {"patient_id": "VT001", "timestamp": "2026-08-01T00:00:00", "glucose_mgdl": 120.0,
         "heart_rate_bpm": 70.0, "hrv_ms": 50.0, "steps": 0, "activity_state": "sleep",
         "is_asleep": True, "data_source": "synthetic"},
        {"patient_id": "VT001", "timestamp": "2026-08-01T00:05:00", "glucose_mgdl": 122.0,
         "heart_rate_bpm": 71.0, "hrv_ms": 51.0, "steps": 0, "activity_state": "sleep",
         "is_asleep": True, "data_source": "synthetic"},
        {"patient_id": "VT001", "timestamp": "2026-08-01T00:10:00", "glucose_mgdl": 121.0,
         "heart_rate_bpm": 70.5, "hrv_ms": 50.5, "steps": 0, "activity_state": "sleep",
         "is_asleep": True, "data_source": "synthetic"},
    ]


def test_clean_sensor_stream_removes_exact_duplicate_timestamp():
    rows = _base_rows() + [_base_rows()[0]]  # exact duplicate (patient_id, timestamp)
    df = pd.DataFrame(rows)
    cleaned, report = clean_sensor_stream(df)
    assert report.duplicates_removed == 1
    assert cleaned.duplicated(subset=["patient_id", "timestamp"]).sum() == 0


def test_clean_sensor_stream_resorts_out_of_order_rows():
    rows = _base_rows()
    rows[0], rows[2] = rows[2], rows[0]  # shuffle
    df = pd.DataFrame(rows)
    cleaned, report = clean_sensor_stream(df)
    assert report.out_of_order_rows_resorted >= 1
    ts = pd.to_datetime(cleaned["timestamp"]).to_numpy()
    assert (np.diff(ts) >= np.timedelta64(0)).all()


def test_clean_sensor_stream_clips_impossible_values():
    rows = _base_rows()
    rows[1]["glucose_mgdl"] = 9999.0  # sensor-error-scale value
    df = pd.DataFrame(rows)
    cleaned, report = clean_sensor_stream(df)
    assert report.out_of_bounds_clipped >= 1
    assert cleaned["glucose_mgdl"].max() <= 400.0


def test_clean_sensor_stream_interpolates_short_gap():
    rows = _base_rows()
    rows[1]["glucose_mgdl"] = np.nan  # single missing reading, 5-min gap
    df = pd.DataFrame(rows)
    cleaned, report = clean_sensor_stream(df)
    assert cleaned["glucose_mgdl"].isna().sum() == 0  # short gap, interpolated
    assert report.missing_values_after == 0


def test_clean_sensor_stream_drops_unparseable_timestamp():
    rows = _base_rows()
    rows[1]["timestamp"] = "not-a-timestamp"
    df = pd.DataFrame(rows)
    cleaned, _ = clean_sensor_stream(df)
    assert len(cleaned) == 2


def test_build_fused_timeline_raises_on_orphan_patient():
    patients = pd.DataFrame([{"patient_id": "VT001", "age": 50, "sex": "F", "bmi": 28.0}])
    ehr = pd.DataFrame([{"patient_id": "VT001", "diagnosis": "T2D", "diabetes_duration_years": 2,
                          "hba1c_pct": 7.0, "medication": "metformin"}])
    sensor = pd.DataFrame(_base_rows())
    sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])
    sensor.loc[0, "patient_id"] = "VT999"  # no static record exists for this patient
    with pytest.raises(ValueError, match="no static record"):
        build_fused_timeline(patients, ehr, sensor)


def test_build_fused_timeline_joins_static_onto_every_row():
    patients = pd.DataFrame([{"patient_id": "VT001", "age": 50, "sex": "F", "bmi": 28.0}])
    ehr = pd.DataFrame([{"patient_id": "VT001", "diagnosis": "T2D", "diabetes_duration_years": 2,
                          "hba1c_pct": 7.0, "medication": "metformin"}])
    sensor = pd.DataFrame(_base_rows())
    sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])
    fused = build_fused_timeline(patients, ehr, sensor)
    assert (fused["hba1c_pct"] == 7.0).all()
    assert len(fused) == len(sensor)
