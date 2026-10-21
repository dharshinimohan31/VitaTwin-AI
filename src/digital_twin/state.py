"""
VitaTwin AI — Digital Twin state engine (Phase 3).

This is the core object the whole product is organized around: a real,
persisted, per-patient state object that changes when new data arrives —
not a dictionary assembled fresh for each dashboard render. The engine
owns twin creation, incremental updates, derived-metric calculation, and
serializable snapshots; prediction/risk/explanation are computed by the
dedicated modules in src/models, src/digital_twin/risk.py, and
src/explainability, and attached to the snapshot — the twin itself does
not know HOW to predict, only how to hold and expose its own state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.features.engineering import FEATURE_COLUMNS, add_features
from src.config import settings


@dataclass
class VitaTwinState:
    """Serializable snapshot of one patient's Digital Twin at a point in time."""
    patient_id: str
    timestamp: str

    demographics: dict[str, Any] = field(default_factory=dict)
    historical: dict[str, Any] = field(default_factory=dict)
    dynamic: dict[str, Any] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)

    prediction: Optional[dict[str, Any]] = None
    risk: Optional[dict[str, Any]] = None
    explanation: Optional[dict[str, Any]] = None

    def snapshot(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dict (deep-copied)."""
        return copy.deepcopy(asdict(self))

    def features_vector(self) -> dict[str, float]:
        """Flat {feature_name: value} dict matching FEATURE_COLUMNS, for model input."""
        pools = (self.derived, self.dynamic, self.demographics, self.historical)
        out: dict[str, float] = {}
        for k in FEATURE_COLUMNS:
            for pool in pools:
                if k in pool and pool[k] is not None:
                    out[k] = pool[k]
                    break
        return out


class DigitalTwinEngine:
    """
    Owns the live twin state for one patient and its update/history log.
    One engine instance per patient_id.
    """

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self._state: Optional[VitaTwinState] = None
        self._history: list[VitaTwinState] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create_from_history(cls, patient_id: str, fused_history: pd.DataFrame) -> "DigitalTwinEngine":
        """
        Build a twin for `patient_id` from its full historical fused
        timeline (output of src.preprocessing.pipeline + feature
        engineering applies as rows are replayed). The twin's current
        state becomes the LATEST row; earlier rows seed the engine's
        history log so rolling/derived features at the latest point are
        computed from real preceding context, not a cold start.
        """
        engine = cls(patient_id)
        patient_rows = fused_history[fused_history["patient_id"] == patient_id].sort_values("timestamp")
        if patient_rows.empty:
            raise ValueError(f"No historical rows found for patient_id={patient_id}")

        featured = add_features(patient_rows)
        for _, row in featured.iterrows():
            engine._apply_row(row, record_history=True)
        return engine

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, new_reading: dict[str, Any], recent_context: pd.DataFrame) -> VitaTwinState:
        """
        Update the twin with ONE new sensor reading.

        `recent_context` must be the fused (patient-filtered, not yet
        featured) history INCLUDING this new reading as the last row —
        rolling features need the preceding window to compute correctly.
        This mirrors exactly how a real streaming update would be fed:
        append the new reading to the buffer, recompute features for the
        newest point.
        """
        featured = add_features(recent_context)
        latest = featured.iloc[-1]
        self._apply_row(latest, record_history=True)
        return self._state

    def _apply_row(self, row: pd.Series, record_history: bool) -> None:
        state = VitaTwinState(
            patient_id=self.patient_id,
            timestamp=str(row["timestamp"]),
            demographics={
                "age": _safe(row.get("age")),
                "sex": row.get("sex"),
                "bmi": _safe(row.get("bmi")),
            },
            historical={
                "diagnosis": row.get("diagnosis"),
                "diabetes_duration_years": _safe(row.get("diabetes_duration_years")),
                "hba1c_pct": _safe(row.get("hba1c_pct")),
                "medication": row.get("medication"),
            },
            dynamic={
                "glucose_mgdl": _safe(row.get("glucose_mgdl")),
                "heart_rate_bpm": _safe(row.get("heart_rate_bpm")),
                "hrv_ms": _safe(row.get("hrv_ms")),
                "steps": _safe(row.get("steps")),
                "activity_state": row.get("activity_state"),
                "is_asleep": bool(row.get("is_asleep", False)),
            },
            derived={k: _safe(row.get(k)) for k in FEATURE_COLUMNS if k not in
                     ("glucose_mgdl", "heart_rate_bpm", "hrv_ms", "age", "bmi", "hba1c_pct",
                      "diabetes_duration_years")},
        )
        self._state = state
        if record_history:
            self._history.append(state)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError(f"Twin for {self.patient_id} has not been created yet.")
        return self._state.snapshot()

    def current_state(self) -> VitaTwinState:
        if self._state is None:
            raise RuntimeError(f"Twin for {self.patient_id} has not been created yet.")
        return self._state

    def history(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        hist = self._history[-limit:] if limit else self._history
        return [s.snapshot() for s in hist]

    def fork(self) -> "DigitalTwinEngine":
        """
        Return a deep-copied engine for What-If simulation. Mutating the
        fork (via its own ._state) can NEVER affect this engine's state —
        this is what src/simulation/whatif.py relies on and
        tests/test_simulation.py verifies explicitly.
        """
        forked = DigitalTwinEngine(self.patient_id)
        forked._state = copy.deepcopy(self._state)
        forked._history = copy.deepcopy(self._history)
        return forked


def _safe(value: Any) -> Any:
    """Convert numpy/pandas scalar types to plain Python for clean JSON serialization."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value
