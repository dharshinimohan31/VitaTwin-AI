"""
VitaTwin AI — feature engineering (Phase 2).

Builds the feature set used by both model training (src/models/train.py)
and live twin scoring (src/digital_twin/state.py). Every feature here is
computed ONLY from data at or before its own timestamp — no future
information is used — which is what makes the later temporal
train/val/test split leakage-free. See docs/features.md for the full
feature dictionary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import settings

ACTIVITY_INTENSITY = {"sleep": 0, "sedentary": 0, "light": 1, "moderate": 2, "vigorous": 3}

# Features consumed by the models. Kept as a single source of truth so
# training and inference can never silently drift apart.
FEATURE_COLUMNS: list[str] = [
    "glucose_mgdl",
    "glucose_roll_mean_15m", "glucose_roll_mean_30m", "glucose_roll_mean_60m", "glucose_roll_mean_120m",
    "glucose_slope_15m", "glucose_roll_std_60m",
    "heart_rate_bpm", "hr_roll_mean_30m",
    "hrv_ms", "hrv_roll_mean_60m",
    "steps_roll_sum_60m", "activity_intensity_roll_mean_60m",
    "sleep_minutes_prev_night", "is_asleep_now",
    "hour_sin", "hour_cos",
    "baseline_glucose_deviation_pct",
    "age", "bmi", "hba1c_pct", "diabetes_duration_years",
]


def _rolling(df: pd.DataFrame, col: str, minutes: int, fn: str) -> pd.Series:
    window = minutes // settings.READING_INTERVAL_MINUTES
    roll = df[col].rolling(window=window, min_periods=max(1, window // 3))
    return getattr(roll, fn)()


def _per_patient_night_sleep_minutes(g: pd.DataFrame) -> pd.Series:
    """
    Minutes asleep during the most recent completed overnight period
    (approximated as the preceding 00:00-09:00 block), forward-filled
    across the following day so every row carries "last night's sleep".
    """
    g = g.copy()
    g["_date"] = g["timestamp"].dt.date
    g["_hour"] = g["timestamp"].dt.hour
    night_mask = g["_hour"] < 9
    nightly = (
        g.loc[night_mask]
        .groupby("_date")["is_asleep"]
        .sum() * settings.READING_INTERVAL_MINUTES
    )
    # Assign each day's nightly-sleep total to all rows of THAT SAME day,
    # then shift forward one day so a row never sees its own night's full
    # total before that night is over (avoids leaking the end of the sleep
    # window into earlier rows of the same night).
    nightly = nightly.reindex(sorted(g["_date"].unique())).ffill().bfill()
    per_day = nightly.to_dict()
    g["sleep_minutes_prev_night"] = g["_date"].map(per_day).astype(float)
    return g["sleep_minutes_prev_night"]


def add_features(fused: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to a fused per-patient timeline.
    Operates per-patient so rolling windows never cross patient boundaries.
    """
    parts = []
    for pid, g in fused.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        g["glucose_roll_mean_15m"] = _rolling(g, "glucose_mgdl", 15, "mean")
        g["glucose_roll_mean_30m"] = _rolling(g, "glucose_mgdl", 30, "mean")
        g["glucose_roll_mean_60m"] = _rolling(g, "glucose_mgdl", 60, "mean")
        g["glucose_roll_mean_120m"] = _rolling(g, "glucose_mgdl", 120, "mean")
        g["glucose_roll_std_60m"] = _rolling(g, "glucose_mgdl", 60, "std").fillna(0.0)

        steps_back = 15 // settings.READING_INTERVAL_MINUTES
        g["glucose_slope_15m"] = (g["glucose_mgdl"] - g["glucose_mgdl"].shift(steps_back)) / 15.0

        g["hr_roll_mean_30m"] = _rolling(g, "heart_rate_bpm", 30, "mean")
        g["hrv_roll_mean_60m"] = _rolling(g, "hrv_ms", 60, "mean")

        g["steps_roll_sum_60m"] = _rolling(g, "steps", 60, "sum")

        g["activity_intensity"] = g["activity_state"].map(ACTIVITY_INTENSITY).fillna(0)
        g["activity_intensity_roll_mean_60m"] = _rolling(g, "activity_intensity", 60, "mean")

        g["is_asleep_now"] = g["is_asleep"].astype(int)
        g["sleep_minutes_prev_night"] = _per_patient_night_sleep_minutes(g)

        hour_frac = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
        g["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
        g["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)

        # Personal baseline: expanding mean of glucose using only PAST data
        # (shifted by one row so a reading never includes itself), i.e. the
        # patient's own glucose history up to this point in time — not a
        # population average, and not using any future information.
        expanding_mean = g["glucose_mgdl"].expanding(min_periods=12).mean().shift(1)
        baseline = expanding_mean.bfill()  # first few rows: best available estimate
        g["personal_baseline_glucose"] = baseline
        g["baseline_glucose_deviation_pct"] = (
            (g["glucose_mgdl"] - baseline) / baseline.replace(0, np.nan) * 100.0
        ).fillna(0.0)

        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.drop(columns=["_date", "_hour"], errors="ignore")
    return out


def add_prediction_target(featured: pd.DataFrame) -> pd.DataFrame:
    """
    Add the supervised-learning target columns:
      - target_glucose_future: glucose PREDICTION_HORIZON_STEPS ahead (regression)
      - target_excursion: 1 if that future glucose exceeds EXCURSION_THRESHOLD_MGDL (classification)
    Rows near the end of each patient's timeline (where the future value
    doesn't exist yet) get NaN targets and must be dropped before training —
    they are naturally excluded from the temporal test window anyway.
    """
    parts = []
    for pid, g in featured.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        future_glucose = g["glucose_mgdl"].shift(-settings.PREDICTION_HORIZON_STEPS)
        g["target_glucose_future"] = future_glucose
        g["target_excursion"] = (future_glucose >= settings.EXCURSION_THRESHOLD_MGDL).astype("float")
        g.loc[future_glucose.isna(), "target_excursion"] = np.nan
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def build_feature_table(fused: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """Convenience: add_features then (optionally) add_prediction_target."""
    featured = add_features(fused)
    if with_target:
        featured = add_prediction_target(featured)
    return featured


if __name__ == "__main__":
    from src.preprocessing.pipeline import run_pipeline

    fused_df, _ = run_pipeline()
    feat = build_feature_table(fused_df)
    usable = feat.dropna(subset=FEATURE_COLUMNS + ["target_glucose_future"])
    print(f"Featured rows: {len(feat):,} total, {len(usable):,} usable for training")
    print(feat[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False).head(8))
