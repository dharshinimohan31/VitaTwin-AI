"""
Central configuration for VitaTwin AI.

All paths are resolved relative to the project root so the codebase runs
identically regardless of the machine or working directory it is invoked
from. No absolute paths are hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR: Path = DATA_DIR / "synthetic"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED: int = 42


# ---------------------------------------------------------------------------
# Synthetic data generation parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticDataConfig:
    """
    Parameters controlling synthetic patient and physiological-stream
    generation.

    IMPORTANT — data provenance:
    The dynamic (wearable/CGM) stream produced by this configuration is a
    RULE-BASED PHYSIOLOGICAL SIMULATION, not a resampling of real patient
    data. It is designed to be schema-compatible with real datasets such as
    OhioT1DM / D1NAMO so a real dynamic stream can be substituted later
    (see src/data/generate_data.py::load_real_dynamic_stream, currently a
    stub). See docs/dataset.md for full disclosure.
    """

    n_patients: int = 8
    n_days: int = 14
    reading_interval_minutes: int = 5
    start_date: str = "2026-08-01"

    # Static/EHR ranges (Type 2 diabetes-relevant adult population)
    age_range: tuple[int, int] = (35, 72)
    bmi_range: tuple[float, float] = (23.0, 39.0)
    hba1c_range: tuple[float, float] = (5.8, 9.8)
    diabetes_duration_range: tuple[int, int] = (0, 18)  # years; 0 = pre-diabetic/newly diagnosed
    medications: tuple[str, ...] = (
        "none",
        "metformin",
        "metformin+sulfonylurea",
        "insulin",
    )

    # Glucose model parameters (mg/dL)
    glucose_baseline_range: tuple[int, int] = (95, 145)
    glucose_meal_amplitude_range: tuple[int, int] = (35, 95)
    glucose_noise_sd: float = 4.0
    meals_per_day: tuple[int, int] = (2, 4)

    # Heart rate / HRV parameters
    resting_hr_range: tuple[int, int] = (58, 88)
    hr_noise_sd: float = 2.5
    hrv_baseline_range: tuple[int, int] = (25, 65)  # ms, rMSSD-like proxy

    # Sleep parameters
    sleep_minutes_mean_range: tuple[int, int] = (300, 460)
    sleep_minutes_sd: float = 35.0


DEFAULT_SYNTHETIC_CONFIG = SyntheticDataConfig()


# ---------------------------------------------------------------------------
# Prediction task definition (Phase 2+)
# ---------------------------------------------------------------------------
READING_INTERVAL_MINUTES: int = DEFAULT_SYNTHETIC_CONFIG.reading_interval_minutes
PREDICTION_HORIZON_MINUTES: int = 120  # predict 2 hours ahead
PREDICTION_HORIZON_STEPS: int = PREDICTION_HORIZON_MINUTES // READING_INTERVAL_MINUTES  # 24 steps

# Hyperglycemic-excursion classification threshold. This is a commonly-used
# clinical reference point (ADA post-meal/general hyperglycemia marker), used
# here ONLY to label a synthetic-data classification task — not a diagnostic
# claim. See docs/model.md for full documentation of this choice.
EXCURSION_THRESHOLD_MGDL: float = 180.0

# Risk category thresholds, applied to the model's predicted excursion
# probability. Documented, not arbitrary: tertile-style split chosen so all
# three categories are reachable and meaningfully separated for a demo-scale
# dataset. See docs/model.md §Risk Thresholds for the full rationale.
RISK_LOW_MAX_PROB: float = 0.33
RISK_MODERATE_MAX_PROB: float = 0.66
# probability >= RISK_MODERATE_MAX_PROB => HIGH

# Feature engineering windows (in 5-minute steps)
ROLLING_WINDOWS_MINUTES: tuple[int, ...] = (15, 30, 60, 120)

# Minimum history required before a twin can be scored (avoids cold-start
# predictions on noise — first ~2 hours of a patient's stream).
MIN_HISTORY_STEPS: int = PREDICTION_HORIZON_STEPS

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
DB_PATH: Path = PROCESSED_DATA_DIR / "vitatwin.db"

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
REGRESSOR_PATH: Path = MODELS_DIR / "glucose_regressor.joblib"
CLASSIFIER_PATH: Path = MODELS_DIR / "excursion_classifier.joblib"
FEATURE_LIST_PATH: Path = MODELS_DIR / "feature_columns.json"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"
