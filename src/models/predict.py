"""
VitaTwin AI — inference (Phase 4-5 / used by digital_twin, simulation, api).

Thin wrapper that loads the trained artifacts saved by src/models/train.py
and scores a single feature vector (as produced by
VitaTwinState.features_vector()). Raises a clear error if models haven't
been trained yet, rather than silently returning a fake prediction.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from src.config import settings


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_artifacts():
    if not (settings.REGRESSOR_PATH.exists() and settings.CLASSIFIER_PATH.exists()):
        raise ModelNotTrainedError(
            "No trained model found. Run `python -m src.models.train` first."
        )
    regressor = joblib.load(settings.REGRESSOR_PATH)
    classifier = joblib.load(settings.CLASSIFIER_PATH)
    feature_columns = json.loads(settings.FEATURE_LIST_PATH.read_text())
    return regressor, classifier, feature_columns


def predict(features: dict[str, Any]) -> dict[str, Any]:
    """
    Score one feature vector. `features` must contain (at least) every key
    in the saved feature_columns list — missing keys raise, rather than
    silently defaulting, since a silent default would be a fabricated
    input masquerading as real data.
    """
    regressor, classifier, feature_columns = _load_artifacts()

    missing = [c for c in feature_columns if c not in features or features[c] is None]
    if missing:
        raise ValueError(f"Cannot predict — missing required features: {missing}")

    row = pd.DataFrame([{c: features[c] for c in feature_columns}])
    predicted_glucose = float(regressor.predict(row)[0])
    excursion_probability = float(classifier.predict_proba(row)[0, 1])

    return {
        "predicted_glucose_mgdl": round(predicted_glucose, 1),
        "horizon_minutes": settings.PREDICTION_HORIZON_MINUTES,
        "excursion_probability": round(excursion_probability, 4),
        "excursion_threshold_mgdl": settings.EXCURSION_THRESHOLD_MGDL,
    }


def reload_artifacts() -> None:
    """Clear the cached models (e.g. after retraining in the same process)."""
    _load_artifacts.cache_clear()
