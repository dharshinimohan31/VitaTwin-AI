"""
VitaTwin AI — SQLite persistence (Phase 3/8).

Schema covers exactly what the implemented system produces: patients,
sensor observations, twin-state snapshots, predictions, and simulation
runs. No speculative tables for unimplemented features.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from src.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER,
    sex TEXT,
    bmi REAL,
    diagnosis TEXT,
    diabetes_duration_years REAL,
    hba1c_pct REAL,
    medication TEXT,
    data_source TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    glucose_mgdl REAL,
    heart_rate_bpm REAL,
    hrv_ms REAL,
    steps INTEGER,
    activity_state TEXT,
    is_asleep INTEGER,
    data_source TEXT,
    UNIQUE(patient_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_observations_patient_ts ON observations(patient_id, timestamp);

CREATE TABLE IF NOT EXISTS twin_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_twin_states_patient_ts ON twin_states(patient_id, timestamp);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    predicted_glucose_mgdl REAL,
    excursion_probability REAL,
    risk_category TEXT,
    horizon_minutes INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_predictions_patient_ts ON predictions(patient_id, timestamp);

CREATE TABLE IF NOT EXISTS simulation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    overrides_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def upsert_patient(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO patients (patient_id, age, sex, bmi, diagnosis, diabetes_duration_years,
                                  hba1c_pct, medication, data_source)
           VALUES (:patient_id, :age, :sex, :bmi, :diagnosis, :diabetes_duration_years,
                   :hba1c_pct, :medication, :data_source)
           ON CONFLICT(patient_id) DO UPDATE SET
             age=excluded.age, sex=excluded.sex, bmi=excluded.bmi, diagnosis=excluded.diagnosis,
             diabetes_duration_years=excluded.diabetes_duration_years, hba1c_pct=excluded.hba1c_pct,
             medication=excluded.medication, data_source=excluded.data_source""",
        row,
    )


def bulk_insert_observations(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    cur = conn.executemany(
        """INSERT OR IGNORE INTO observations
           (patient_id, timestamp, glucose_mgdl, heart_rate_bpm, hrv_ms, steps,
            activity_state, is_asleep, data_source)
           VALUES (:patient_id, :timestamp, :glucose_mgdl, :heart_rate_bpm, :hrv_ms, :steps,
                   :activity_state, :is_asleep, :data_source)""",
        rows,
    )
    return cur.rowcount


def save_twin_state(conn: sqlite3.Connection, patient_id: str, timestamp: str, state: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO twin_states (patient_id, timestamp, state_json) VALUES (?, ?, ?)",
        (patient_id, timestamp, json.dumps(state)),
    )


def save_prediction(conn: sqlite3.Connection, patient_id: str, timestamp: str, prediction: dict, risk: dict) -> None:
    conn.execute(
        """INSERT INTO predictions
           (patient_id, timestamp, predicted_glucose_mgdl, excursion_probability, risk_category, horizon_minutes)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (patient_id, timestamp, prediction["predicted_glucose_mgdl"], prediction["excursion_probability"],
         risk["risk_category"], prediction["horizon_minutes"]),
    )


def save_simulation_run(conn: sqlite3.Connection, patient_id: str, timestamp: str, overrides: dict, result: dict) -> None:
    conn.execute(
        "INSERT INTO simulation_runs (patient_id, timestamp, overrides_json, result_json) VALUES (?, ?, ?, ?)",
        (patient_id, timestamp, json.dumps(overrides), json.dumps(result)),
    )


def seed_from_synthetic_data() -> dict[str, int]:
    """Populate patients + observations tables from the generated CSVs."""
    import pandas as pd

    patients = pd.read_csv(settings.SYNTHETIC_DATA_DIR / "patients.csv")
    ehr = pd.read_csv(settings.SYNTHETIC_DATA_DIR / "ehr_records.csv")
    sensor = pd.read_csv(settings.SYNTHETIC_DATA_DIR / "sensor_readings.csv")
    merged = patients.merge(ehr.drop(columns=["data_source"]), on="patient_id", how="left")

    init_db()
    n_obs = 0
    with get_connection() as conn:
        for _, row in merged.iterrows():
            upsert_patient(conn, row.to_dict())
        obs_rows = sensor.assign(is_asleep=sensor["is_asleep"].astype(int)).to_dict("records")
        n_obs = bulk_insert_observations(conn, obs_rows)
    return {"patients": len(merged), "observations_inserted": n_obs}


if __name__ == "__main__":
    stats = seed_from_synthetic_data()
    print(f"DB initialized at {settings.DB_PATH}")
    print(stats)
