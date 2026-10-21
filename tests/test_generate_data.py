"""
Sanity and reproducibility tests for the Phase 1 synthetic data generator.

Run with:  pytest tests/test_generate_data.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import SyntheticDataConfig
from src.data.generate_data import generate_all, generate_patients_and_ehr


@pytest.fixture(scope="module")
def small_cfg() -> SyntheticDataConfig:
    return SyntheticDataConfig(n_patients=3, n_days=3)


@pytest.fixture(scope="module")
def generated(tmp_path_factory, small_cfg) -> dict[str, pd.DataFrame]:
    out_dir = tmp_path_factory.mktemp("synthetic")
    paths = generate_all(cfg=small_cfg, seed=42, out_dir=out_dir)
    return {name: pd.read_csv(path) for name, path in paths.items()}


def test_patient_count(generated, small_cfg):
    assert len(generated["patients"]) == small_cfg.n_patients
    assert len(generated["ehr_records"]) == small_cfg.n_patients


def test_expected_row_count(generated, small_cfg):
    expected_per_patient = 24 * 60 // small_cfg.reading_interval_minutes * small_cfg.n_days
    expected_total = expected_per_patient * small_cfg.n_patients
    assert len(generated["sensor_readings"]) == expected_total


def test_no_duplicate_timestamps_per_patient(generated):
    df = generated["sensor_readings"]
    for patient_id, group in df.groupby("patient_id"):
        assert not group["timestamp"].duplicated().any(), f"duplicate timestamps for {patient_id}"


def test_timestamps_monotonic_per_patient(generated):
    df = generated["sensor_readings"]
    for patient_id, group in df.groupby("patient_id"):
        ts = pd.to_datetime(group["timestamp"])
        assert ts.is_monotonic_increasing, f"non-monotonic timestamps for {patient_id}"


def test_no_missing_values(generated):
    for name, df in generated.items():
        assert not df.isna().any().any(), f"missing values found in {name}"


def test_glucose_within_physiological_bounds(generated):
    glucose = generated["sensor_readings"]["glucose_mgdl"]
    assert glucose.min() >= 55
    assert glucose.max() <= 400
    # No systematic clipping at the hard bound -- would indicate a modeling bug.
    assert (glucose >= 399).mean() < 0.01


def test_heart_rate_within_physiological_bounds(generated):
    hr = generated["sensor_readings"]["heart_rate_bpm"]
    assert hr.min() >= 40
    assert hr.max() <= 190


def test_all_rows_tagged_with_data_source(generated):
    for name, df in generated.items():
        assert (df["data_source"] == "synthetic").all(), f"missing/incorrect data_source tag in {name}"


def test_reproducibility_same_seed(small_cfg, tmp_path):
    """Same seed must produce byte-identical numeric output."""
    _, ehr_a, profiles_a = generate_patients_and_ehr(small_cfg, seed=123)
    _, ehr_b, profiles_b = generate_patients_and_ehr(small_cfg, seed=123)
    pd.testing.assert_frame_equal(ehr_a, ehr_b)
    assert [p.hba1c for p in profiles_a] == [p.hba1c for p in profiles_b]


def test_different_seeds_differ(small_cfg):
    """Different seeds should (almost certainly) produce different patients."""
    _, ehr_a, _ = generate_patients_and_ehr(small_cfg, seed=1)
    _, ehr_b, _ = generate_patients_and_ehr(small_cfg, seed=2)
    assert not ehr_a["hba1c_pct"].equals(ehr_b["hba1c_pct"])


def test_patient_ids_match_across_tables(generated):
    patient_ids = set(generated["patients"]["patient_id"])
    assert set(generated["ehr_records"]["patient_id"]) == patient_ids
    assert set(generated["sensor_readings"]["patient_id"]) == patient_ids
