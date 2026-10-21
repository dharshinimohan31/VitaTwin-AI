"""
VitaTwin AI — data preprocessing & fusion pipeline (Phase 2).

Takes the raw synthetic patients/ehr/sensor CSVs and produces a clean,
temporally-sorted, deduplicated, fused timeline per patient, ready for
feature engineering. Handles the realistic messiness a real pipeline must
handle even though data generation itself (Phase 1) outputs clean rows:
duplicate timestamps, out-of-order rows, and missing values are all
exercised by the test suite using deliberately corrupted inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

REQUIRED_PATIENT_COLS = {"patient_id", "age", "sex", "bmi"}
REQUIRED_EHR_COLS = {"patient_id", "diagnosis", "diabetes_duration_years", "hba1c_pct", "medication"}
REQUIRED_SENSOR_COLS = {
    "patient_id", "timestamp", "glucose_mgdl", "heart_rate_bpm",
    "hrv_ms", "steps", "activity_state", "is_asleep",
}

NUMERIC_SENSOR_BOUNDS = {
    "glucose_mgdl": (40.0, 400.0),
    "heart_rate_bpm": (30.0, 220.0),
    "hrv_ms": (5.0, 200.0),
    "steps": (0, 400),  # per 5-min interval
}


@dataclass
class ValidationReport:
    """Summary of what the pipeline found and fixed, for transparency."""
    rows_in: int
    rows_out: int
    duplicates_removed: int
    out_of_order_rows_resorted: int
    missing_values_before: int
    missing_values_after: int
    out_of_bounds_clipped: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _validate_schema(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def load_raw_tables(
    patients_path=None, ehr_path=None, sensor_path=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw synthetic tables and validate their schema."""
    patients_path = patients_path or settings.SYNTHETIC_DATA_DIR / "patients.csv"
    ehr_path = ehr_path or settings.SYNTHETIC_DATA_DIR / "ehr_records.csv"
    sensor_path = sensor_path or settings.SYNTHETIC_DATA_DIR / "sensor_readings.csv"

    for p in (patients_path, ehr_path, sensor_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run `python -m src.data.generate_data` first."
            )

    patients = pd.read_csv(patients_path)
    ehr = pd.read_csv(ehr_path)
    sensor = pd.read_csv(sensor_path)

    _validate_schema(patients, REQUIRED_PATIENT_COLS, "patients")
    _validate_schema(ehr, REQUIRED_EHR_COLS, "ehr_records")
    _validate_schema(sensor, REQUIRED_SENSOR_COLS, "sensor_readings")

    return patients, ehr, sensor


def clean_sensor_stream(sensor: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Clean a single fused sensor table (all patients) end-to-end:
    parse timestamps, drop exact duplicates, resolve duplicate
    (patient_id, timestamp) pairs, sort chronologically per patient,
    clip physiologically impossible values, and interpolate small gaps.
    """
    rows_in = len(sensor)
    df = sensor.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    missing_before = int(df.isna().sum().sum())

    # Drop rows with unparseable timestamp or missing patient_id outright —
    # these cannot be placed on a timeline at all.
    df = df.dropna(subset=["timestamp", "patient_id"])

    # Duplicate (patient_id, timestamp) pairs: keep the first, drop the rest.
    # (A real device could emit a retransmit with the same timestamp.)
    dup_mask = df.duplicated(subset=["patient_id", "timestamp"], keep="first")
    duplicates_removed = int(dup_mask.sum())
    df = df[~dup_mask]

    # Out-of-order rows: sort per patient by timestamp. We detect how many
    # rows were out of order before sorting, for the report.
    df_check = df.sort_values(["patient_id"], kind="stable")
    out_of_order = 0
    for _, g in df_check.groupby("patient_id", sort=False):
        ts = g["timestamp"].to_numpy()
        out_of_order += int((np.diff(ts) < np.timedelta64(0)).sum())
    df = df.sort_values(["patient_id", "timestamp"], kind="stable").reset_index(drop=True)

    # Clip physiologically impossible values rather than dropping the row
    # outright (we still have HR/steps/etc. to use even if glucose spiked
    # to an impossible sensor-error value).
    clipped = 0
    for col, (lo, hi) in NUMERIC_SENSOR_BOUNDS.items():
        if col not in df.columns:
            continue
        before = df[col].copy()
        df[col] = df[col].clip(lower=lo, upper=hi)
        clipped += int((before != df[col]).sum())

    # Interpolate short gaps (<=3 consecutive missing steps = 15 min) in the
    # core numeric signals per patient; longer gaps are left as NaN (the
    # twin/feature layer must be able to tolerate real missingness).
    numeric_cols = ["glucose_mgdl", "heart_rate_bpm", "hrv_ms", "steps"]
    parts = []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.set_index("timestamp")
        g[numeric_cols] = g[numeric_cols].interpolate(method="time", limit=3, limit_direction="both")
        g = g.reset_index()
        g["patient_id"] = pid
        parts.append(g)
    df = pd.concat(parts, ignore_index=True) if parts else df

    missing_after = int(df[numeric_cols].isna().sum().sum())

    report = ValidationReport(
        rows_in=rows_in,
        rows_out=len(df),
        duplicates_removed=duplicates_removed,
        out_of_order_rows_resorted=out_of_order,
        missing_values_before=missing_before,
        missing_values_after=missing_after,
        out_of_bounds_clipped=clipped,
    )
    logger.info("Sensor cleaning report: %s", report.as_dict())
    return df, report


def build_fused_timeline(
    patients: pd.DataFrame, ehr: pd.DataFrame, sensor: pd.DataFrame
) -> pd.DataFrame:
    """
    Join cleaned sensor data with static patient/EHR context, broadcasting
    the (slow-changing) static fields onto every dynamic row for that
    patient. One row per (patient_id, timestamp), ready for feature
    engineering.
    """
    static = patients.merge(ehr.drop(columns=["data_source"], errors="ignore"), on="patient_id", how="left")
    fused = sensor.merge(static, on="patient_id", how="left", suffixes=("", "_static"))
    if fused["age"].isna().any():
        orphans = sorted(fused.loc[fused["age"].isna(), "patient_id"].unique())
        raise ValueError(f"Sensor rows reference patient_id(s) with no static record: {orphans}")
    return fused.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def run_pipeline(
    patients_path=None, ehr_path=None, sensor_path=None
) -> tuple[pd.DataFrame, ValidationReport]:
    """End-to-end: load raw tables, clean, fuse. Returns (fused_df, report)."""
    patients, ehr, sensor = load_raw_tables(patients_path, ehr_path, sensor_path)
    clean_sensor, report = clean_sensor_stream(sensor)
    fused = build_fused_timeline(patients, ehr, clean_sensor)
    return fused, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fused_df, rpt = run_pipeline()
    print(f"Fused timeline: {len(fused_df):,} rows, {fused_df['patient_id'].nunique()} patients")
    print("Validation report:", rpt.as_dict())
