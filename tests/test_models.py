import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.models.train import temporal_split

MODEL_TRAINED = settings.REGRESSOR_PATH.exists() and settings.CLASSIFIER_PATH.exists()


def test_temporal_split_preserves_chronological_order_per_patient():
    ts = pd.date_range("2026-08-01", periods=20, freq="5min")
    df = pd.DataFrame({"patient_id": "VT001", "timestamp": ts, "value": range(20)})
    train, val, test = temporal_split(df)
    assert train["value"].max() < val["value"].min()
    assert val["value"].max() < test["value"].min()


def test_temporal_split_keeps_patients_separate():
    ts = pd.date_range("2026-08-01", periods=10, freq="5min")
    df1 = pd.DataFrame({"patient_id": "VT001", "timestamp": ts, "value": range(10)})
    df2 = pd.DataFrame({"patient_id": "VT002", "timestamp": ts, "value": range(100, 110)})
    combined = pd.concat([df1, df2], ignore_index=True)
    train, val, test = temporal_split(combined)
    assert set(train["patient_id"]) == {"VT001", "VT002"}
    # each patient contributes to each split (70/15/15 of 10 rows each)
    assert "VT001" in set(test["patient_id"])
    assert "VT002" in set(test["patient_id"])


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts — run `python -m src.models.train`")
def test_predict_rejects_missing_features():
    from src.models.predict import predict
    with pytest.raises(ValueError, match="missing required features"):
        predict({"glucose_mgdl": 120.0})  # deliberately incomplete


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_predict_returns_expected_shape_and_bounds():
    from src.features.engineering import FEATURE_COLUMNS
    from src.models.predict import predict

    dummy = {c: 1.0 for c in FEATURE_COLUMNS}
    dummy.update({"glucose_mgdl": 140, "glucose_roll_mean_15m": 140, "glucose_roll_mean_30m": 140,
                   "glucose_roll_mean_60m": 140, "glucose_roll_mean_120m": 140, "age": 55, "bmi": 29,
                   "hba1c_pct": 7.5, "diabetes_duration_years": 4})
    result = predict(dummy)
    assert "predicted_glucose_mgdl" in result
    assert "excursion_probability" in result
    assert 0.0 <= result["excursion_probability"] <= 1.0
    assert result["horizon_minutes"] == settings.PREDICTION_HORIZON_MINUTES


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_predict_is_deterministic_for_same_input():
    from src.features.engineering import FEATURE_COLUMNS
    from src.models.predict import predict

    dummy = {c: 2.0 for c in FEATURE_COLUMNS}
    r1 = predict(dummy)
    r2 = predict(dummy)
    assert r1 == r2


@pytest.mark.skipif(not MODEL_TRAINED, reason="requires trained model artifacts")
def test_saved_metrics_file_has_real_values_not_placeholders():
    import json
    metrics = json.loads(settings.METRICS_PATH.read_text())
    assert metrics["regression"]["mae"] > 0
    assert metrics["regression"]["n"] > 0
    assert 0 <= metrics["classification"]["roc_auc"] <= 1
    assert metrics["classification"]["n"] > 0
