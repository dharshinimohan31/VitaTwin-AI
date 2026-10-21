import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.features.engineering import FEATURE_COLUMNS, add_features, add_prediction_target


def _synthetic_patient_df(n_steps=40, start="2026-08-01T00:00:00"):
    ts = pd.date_range(start, periods=n_steps, freq="5min")
    rng = np.random.default_rng(0)
    glucose = 120 + np.cumsum(rng.normal(0, 2, n_steps))
    return pd.DataFrame({
        "patient_id": "VT001",
        "timestamp": ts,
        "glucose_mgdl": glucose,
        "heart_rate_bpm": 70 + rng.normal(0, 1, n_steps),
        "hrv_ms": 50 + rng.normal(0, 1, n_steps),
        "steps": rng.integers(0, 50, n_steps),
        "activity_state": "light",
        "is_asleep": False,
        "age": 50, "sex": "F", "bmi": 28.0,
        "diagnosis": "T2D", "diabetes_duration_years": 2, "hba1c_pct": 7.0, "medication": "metformin",
    })


def test_add_features_produces_all_declared_columns():
    df = _synthetic_patient_df()
    featured = add_features(df)
    for col in FEATURE_COLUMNS:
        assert col in featured.columns, f"missing feature column {col}"


def test_rolling_features_use_only_past_data_no_leakage():
    """
    Changing a FUTURE glucose value must not change an EARLIER row's
    rolling features — that would be leakage.
    """
    df = _synthetic_patient_df()
    featured_before = add_features(df.copy())

    df_mutated = df.copy()
    df_mutated.loc[df_mutated.index[-1], "glucose_mgdl"] = 999.0  # mutate only the LAST row
    featured_after = add_features(df_mutated)

    # every row except the mutated last one (and rows whose window includes it)
    # must be identical — check the first 20 rows, far from the mutation
    pd.testing.assert_series_equal(
        featured_before.loc[:20, "glucose_roll_mean_60m"],
        featured_after.loc[:20, "glucose_roll_mean_60m"],
    )


def test_prediction_target_shifts_correctly():
    df = _synthetic_patient_df(n_steps=40)
    featured = add_features(df)
    targeted = add_prediction_target(featured)

    horizon = settings.PREDICTION_HORIZON_STEPS
    # target at row i should equal glucose_mgdl at row i+horizon
    for i in range(0, len(targeted) - horizon):
        assert targeted.loc[i, "target_glucose_future"] == pytest.approx(
            targeted.loc[i + horizon, "glucose_mgdl"]
        )

    # rows too close to the end must have NaN target (nothing to predict)
    assert targeted["target_glucose_future"].iloc[-1:].isna().all()


def test_excursion_target_matches_threshold():
    df = _synthetic_patient_df(n_steps=40)
    df.loc[df.index[-5:], "glucose_mgdl"] = 250.0  # force a clear excursion near the end
    featured = add_features(df)
    targeted = add_prediction_target(featured)
    horizon = settings.PREDICTION_HORIZON_STEPS
    row = targeted.iloc[0]
    if pd.notna(row["target_glucose_future"]):
        expected = float(row["target_glucose_future"] >= settings.EXCURSION_THRESHOLD_MGDL)
        assert row["target_excursion"] == expected


def test_rolling_windows_never_cross_patient_boundary():
    df1 = _synthetic_patient_df(n_steps=10, start="2026-08-01T00:00:00")
    df2 = _synthetic_patient_df(n_steps=10, start="2026-08-01T00:00:00")
    df2["patient_id"] = "VT002"
    df2["glucose_mgdl"] = 300.0  # wildly different patient
    combined = pd.concat([df1, df2], ignore_index=True)
    featured = add_features(combined)
    # VT001's first row rolling mean must reflect ONLY VT001's own data
    first_vt001 = featured[featured.patient_id == "VT001"].iloc[0]
    assert first_vt001["glucose_roll_mean_15m"] < 200  # not polluted by VT002's 300s
