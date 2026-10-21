"""
VitaTwin AI — Synthetic Data Generator (Phase 1)

Generates a reproducible, schema-compatible demonstration dataset:

  - patients.csv         : static demographics
  - ehr_records.csv      : static/historical medical record per patient
  - sensor_readings.csv  : dynamic wearable/CGM time-series per patient

DATA PROVENANCE (read this before using results in any report):
  The dynamic stream (glucose, heart rate, HRV, steps, sleep) is produced by
  a RULE-BASED PHYSIOLOGICAL SIMULATION documented in this file. It is NOT
  resampled from real OhioT1DM/D1NAMO patient data — this sandbox has no
  network access to PhysioNet-hosted datasets. The schema is intentionally
  compatible with those real datasets so `load_real_dynamic_stream()` (a
  documented stub below) can later replace the synthetic generator without
  touching any downstream code. Every row is tagged `data_source=synthetic`
  so this is traceable end-to-end. See docs/dataset.md for full disclosure
  and the physiological assumptions behind the simulation.

Usage:
    python -m src.data.generate_data --n-patients 8 --n-days 14 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import (
    DEFAULT_SYNTHETIC_CONFIG,
    GLOBAL_SEED,
    SYNTHETIC_DATA_DIR,
    SyntheticDataConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("vitatwin.generate_data")

ACTIVITY_STATES = ("sleep", "sedentary", "light", "moderate", "vigorous")

# Step-rate (steps per 5-minute interval) and HR increment (bpm above resting)
# per activity state. These are illustrative, documented assumptions — not
# clinical constants.
ACTIVITY_STEP_RATE = {"sleep": 0, "sedentary": 4, "light": 45, "moderate": 110, "vigorous": 220}
ACTIVITY_HR_DELTA = {"sleep": -8, "sedentary": 0, "light": 12, "moderate": 30, "vigorous": 55}
ACTIVITY_HRV_DELTA = {"sleep": 12, "sedentary": 0, "light": -4, "moderate": -12, "vigorous": -20}


@dataclass
class PatientProfile:
    """Internal generation profile — a superset of what's written to disk."""

    patient_id: str
    age: int
    sex: str
    bmi: float
    hba1c: float
    diabetes_duration: int
    medication: str
    diagnosis: str
    glucose_baseline: float
    insulin_resistance_factor: float  # >1.0 = larger/slower meal responses
    resting_hr: float
    hrv_baseline: float
    sleep_minutes_mean: float
    sleep_start_hour: float  # e.g. 22.5 = 22:30


# ---------------------------------------------------------------------------
# Static / EHR generation
# ---------------------------------------------------------------------------
def _sample_patient_profile(rng: np.random.Generator, cfg: SyntheticDataConfig, idx: int) -> PatientProfile:
    patient_id = f"VT{idx + 1:03d}"
    age = int(rng.integers(*cfg.age_range))
    sex = rng.choice(["M", "F"])
    bmi = round(float(rng.uniform(*cfg.bmi_range)), 1)
    hba1c = round(float(rng.uniform(*cfg.hba1c_range)), 1)
    diabetes_duration = int(rng.integers(*cfg.diabetes_duration_range))
    medication = rng.choice(cfg.medications, p=[0.15, 0.45, 0.25, 0.15])

    diagnosis = "Type 2 Diabetes Mellitus" if hba1c >= 6.5 else "Pre-diabetes (impaired glucose tolerance)"

    # Higher HbA1c and longer duration -> higher baseline & slower meal clearance.
    glucose_baseline = float(rng.uniform(*cfg.glucose_baseline_range)) + (hba1c - 6.0) * 6.0
    insulin_resistance_factor = 1.0 + max(0.0, (hba1c - 6.0)) * 0.18 + (bmi - 25) * 0.01
    insulin_resistance_factor = float(np.clip(insulin_resistance_factor, 0.8, 2.2))

    resting_hr = float(rng.uniform(*cfg.resting_hr_range))
    hrv_baseline = float(rng.uniform(*cfg.hrv_baseline_range))
    sleep_minutes_mean = float(rng.uniform(*cfg.sleep_minutes_mean_range))
    sleep_start_hour = float(rng.uniform(21.5, 24.0))

    return PatientProfile(
        patient_id=patient_id,
        age=age,
        sex=sex,
        bmi=bmi,
        hba1c=hba1c,
        diabetes_duration=diabetes_duration,
        medication=medication,
        diagnosis=diagnosis,
        glucose_baseline=glucose_baseline,
        insulin_resistance_factor=insulin_resistance_factor,
        resting_hr=resting_hr,
        hrv_baseline=hrv_baseline,
        sleep_minutes_mean=sleep_minutes_mean,
        sleep_start_hour=sleep_start_hour,
    )


def generate_patients_and_ehr(
    cfg: SyntheticDataConfig, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[PatientProfile]]:
    """Generate the static patients table and the EHR record table."""
    rng = np.random.default_rng(seed)
    profiles = [_sample_patient_profile(rng, cfg, i) for i in range(cfg.n_patients)]

    patients_df = pd.DataFrame(
        [
            {
                "patient_id": p.patient_id,
                "age": p.age,
                "sex": p.sex,
                "bmi": p.bmi,
                "data_source": "synthetic",
            }
            for p in profiles
        ]
    )

    ehr_df = pd.DataFrame(
        [
            {
                "patient_id": p.patient_id,
                "diagnosis": p.diagnosis,
                "diabetes_duration_years": p.diabetes_duration,
                "hba1c_pct": p.hba1c,
                "medication": p.medication,
                "record_date": cfg.start_date,
                "data_source": "synthetic",
            }
            for p in profiles
        ]
    )
    return patients_df, ehr_df, profiles


# ---------------------------------------------------------------------------
# Dynamic / wearable + CGM stream generation
# ---------------------------------------------------------------------------
def _daily_meal_events(rng: np.random.Generator, cfg: SyntheticDataConfig) -> list[tuple[float, float]]:
    """Return [(hour_of_day, carb_load), ...] for one simulated day."""
    windows = [(7.0, 9.0), (12.0, 14.0), (19.0, 21.0)]
    events = [(float(rng.uniform(*w)), float(rng.uniform(0.6, 1.4))) for w in windows]
    n_meals = int(rng.integers(*cfg.meals_per_day))
    if n_meals > 3:
        events.append((float(rng.uniform(15.5, 17.5)), float(rng.uniform(0.3, 0.8))))  # snack
    return events


def _meal_glucose_effect(minutes_since_meal: float, amplitude: float, resistance: float) -> float:
    """
    Single-peak meal-response kernel: linear rise to exactly `amplitude` at
    t=peak_time, then exponential decay. Peak time and decay lengthen with
    `resistance` (documented proxy for insulin resistance/slower clearance),
    but the kernel is normalized so it never exceeds `amplitude` — avoiding
    unbounded stacking when meals overlap.
    """
    if minutes_since_meal < 0:
        return 0.0
    peak_time = 45.0 * (0.9 + 0.2 * resistance)
    decay = 90.0 * resistance
    if minutes_since_meal <= peak_time:
        return amplitude * (minutes_since_meal / peak_time)
    return amplitude * np.exp(-(minutes_since_meal - peak_time) / decay)


def _activity_state(rng: np.random.Generator, hour: float, sleeping: bool) -> str:
    if sleeping:
        return "sleep"
    # Base probabilities shift with time of day (more activity midday/evening).
    if 6.0 <= hour < 9.0 or 17.0 <= hour < 20.0:
        probs = [0.30, 0.30, 0.25, 0.15]  # sedentary, light, moderate, vigorous
    elif 9.0 <= hour < 17.0:
        probs = [0.45, 0.35, 0.17, 0.03]
    else:
        probs = [0.80, 0.15, 0.05, 0.00]
    return rng.choice(["sedentary", "light", "moderate", "vigorous"], p=probs)


def generate_dynamic_stream_for_patient(
    profile: PatientProfile, cfg: SyntheticDataConfig, seed: int
) -> pd.DataFrame:
    """Generate one patient's full dynamic (wearable + CGM) time-series."""
    rng = np.random.default_rng(seed)
    interval = cfg.reading_interval_minutes
    steps_per_day = int(24 * 60 / interval)
    start = datetime.fromisoformat(cfg.start_date)

    rows: list[dict] = []
    # Recent-activity memory (minutes of moderate+ activity in last hour) speeds meal clearance.
    recent_activity_boost = 0.0

    for day in range(cfg.n_days):
        day_start = start + timedelta(days=day)
        meals = _daily_meal_events(rng, cfg)
        sleep_start = profile.sleep_start_hour % 24
        sleep_minutes = max(180.0, rng.normal(profile.sleep_minutes_mean, 30.0))
        sleep_end = (sleep_start + sleep_minutes / 60.0) % 24

        for step in range(steps_per_day):
            minute_of_day = step * interval
            hour = minute_of_day / 60.0
            timestamp = day_start + timedelta(minutes=minute_of_day)

            if sleep_start < sleep_end:
                sleeping = sleep_start <= hour < sleep_end
            else:  # wraps past midnight
                sleeping = hour >= sleep_start or hour < sleep_end

            state = _activity_state(rng, hour, sleeping)
            recent_activity_boost = 0.7 * recent_activity_boost + 0.3 * (
                1.0 if state in ("moderate", "vigorous") else 0.0
            )

            # --- Glucose ---
            circadian = 8.0 * np.sin((hour - 5.0) / 24.0 * 2 * np.pi)  # mild dawn-phenomenon-like rise
            meal_effect = 0.0
            for meal_hour, carb_load in meals:
                mins_since = minute_of_day - meal_hour * 60.0
                if 0 <= mins_since <= 240:
                    resistance = profile.insulin_resistance_factor * (1 - 0.25 * recent_activity_boost)
                    resistance = max(0.5, resistance)
                    base_amplitude = carb_load * float(rng.uniform(*cfg.glucose_meal_amplitude_range))
                    # Worse glycemic control -> modestly larger spikes, not just slower clearance.
                    amplitude = base_amplitude * (0.75 + 0.25 * resistance)
                    meal_effect += _meal_glucose_effect(mins_since, amplitude=amplitude, resistance=resistance)
            activity_drop = 6.0 * recent_activity_boost
            noise = float(rng.normal(0, cfg.glucose_noise_sd))
            glucose = profile.glucose_baseline + circadian + meal_effect - activity_drop + noise
            glucose = float(np.clip(glucose, 55, 400))

            # --- Heart rate & HRV ---
            hr = profile.resting_hr + ACTIVITY_HR_DELTA[state] + float(rng.normal(0, cfg.hr_noise_sd))
            hr = float(np.clip(hr, 40, 190))
            hrv = profile.hrv_baseline + ACTIVITY_HRV_DELTA[state] + float(rng.normal(0, 3.0))
            hrv = float(np.clip(hrv, 5, 120))

            # --- Steps ---
            steps = max(0, int(rng.poisson(ACTIVITY_STEP_RATE[state])))

            rows.append(
                {
                    "patient_id": profile.patient_id,
                    "timestamp": timestamp.isoformat(),
                    "glucose_mgdl": round(glucose, 1),
                    "heart_rate_bpm": round(hr, 1),
                    "hrv_ms": round(hrv, 1),
                    "steps": steps,
                    "activity_state": state,
                    "is_asleep": bool(sleeping),
                    "data_source": "synthetic",
                }
            )

    return pd.DataFrame(rows)


def load_real_dynamic_stream(source_path: Path) -> pd.DataFrame:  # pragma: no cover - documented stub
    """
    STUB — not implemented in this sandbox (no network access to
    PhysioNet-hosted datasets such as OhioT1DM or D1NAMO).

    When real data is available, implement this to read the raw
    OhioT1DM/D1NAMO export at `source_path` and return a DataFrame with the
    exact same columns as `generate_dynamic_stream_for_patient` (with
    `data_source` set to "real_proxy"), so every downstream module
    (preprocessing, features, digital_twin, models) works unchanged.
    """
    raise NotImplementedError(
        "Real dynamic-stream loading is not available in this environment. "
        "See docs/dataset.md for the disclosed data-provenance strategy."
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all(
    cfg: SyntheticDataConfig = DEFAULT_SYNTHETIC_CONFIG,
    seed: int = GLOBAL_SEED,
    out_dir: Path = SYNTHETIC_DATA_DIR,
) -> dict[str, Path]:
    """Generate the full demonstration dataset and write it to `out_dir`."""
    logger.info(
        "Generating synthetic dataset: n_patients=%d n_days=%d seed=%d",
        cfg.n_patients,
        cfg.n_days,
        seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    patients_df, ehr_df, profiles = generate_patients_and_ehr(cfg, seed)

    dynamic_frames = []
    for i, profile in enumerate(profiles):
        patient_seed = seed + 1000 + i  # distinct, reproducible per-patient stream
        df = generate_dynamic_stream_for_patient(profile, cfg, patient_seed)
        dynamic_frames.append(df)
        logger.info("  %s: %d readings generated", profile.patient_id, len(df))

    sensor_df = pd.concat(dynamic_frames, ignore_index=True)

    patients_path = out_dir / "patients.csv"
    ehr_path = out_dir / "ehr_records.csv"
    sensor_path = out_dir / "sensor_readings.csv"

    patients_df.to_csv(patients_path, index=False)
    ehr_df.to_csv(ehr_path, index=False)
    sensor_df.to_csv(sensor_path, index=False)

    logger.info("Wrote %s (%d rows)", patients_path, len(patients_df))
    logger.info("Wrote %s (%d rows)", ehr_path, len(ehr_df))
    logger.info("Wrote %s (%d rows)", sensor_path, len(sensor_df))

    return {"patients": patients_path, "ehr_records": ehr_path, "sensor_readings": sensor_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VitaTwin AI synthetic demonstration dataset.")
    parser.add_argument("--n-patients", type=int, default=DEFAULT_SYNTHETIC_CONFIG.n_patients)
    parser.add_argument("--n-days", type=int, default=DEFAULT_SYNTHETIC_CONFIG.n_days)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--out-dir", type=Path, default=SYNTHETIC_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = SyntheticDataConfig(n_patients=args.n_patients, n_days=args.n_days)
    generate_all(cfg=cfg, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
