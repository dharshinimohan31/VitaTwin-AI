"""
VitaTwin AI — What-If simulation engine (Phase 7).

Forks the twin's CURRENT feature vector, applies user-chosen overrides to
a small set of model-supported, editable variables, re-runs the REAL
trained model on the modified vector, and returns baseline vs. simulated
side by side. This never touches the real DigitalTwinEngine's state —
callers pass in a feature dict (from engine.current_state().features_vector())
and get a new dict back; nothing is mutated in place.

Only variables the model actually uses, and that are plausibly
patient/clinician-editable, are exposed for simulation — we do not let
someone "simulate" a different HbA1c or age, which would be meaningless.
"""

from __future__ import annotations

import copy
from typing import Any

from src.digital_twin.risk import assess
from src.models.predict import predict

# Variable -> (feature(s) it affects, how). Kept intentionally small and
# honest about what each lever actually changes in the feature vector —
# no hidden multi-feature "magic" adjustments.
SIMULATABLE_VARIABLES = {
    "activity_boost_steps_60m": {
        "label": "Additional steps in the last 60 minutes",
        "affects": ["steps_roll_sum_60m", "activity_intensity_roll_mean_60m"],
        "default": 0, "min": 0, "max": 3000, "step": 250,
    },
    "sleep_adjustment_minutes": {
        "label": "Change in last night's sleep (minutes)",
        "affects": ["sleep_minutes_prev_night"],
        "default": 0, "min": -120, "max": 120, "step": 15,
    },
    "glucose_adjustment_mgdl": {
        "label": "Hypothetical change in current glucose (meal/medication proxy)",
        "affects": ["glucose_mgdl", "glucose_roll_mean_15m", "glucose_roll_mean_30m"],
        "default": 0, "min": -60, "max": 60, "step": 5,
    },
}

DISCLAIMER = (
    "Model-based hypothetical simulation — not a clinical treatment recommendation. "
    "Shows how the trained model's output would change under a hypothetical input; "
    "does not claim that performing this action produces this result in reality."
)


def apply_overrides(features: dict[str, Any], overrides: dict[str, float]) -> dict[str, Any]:
    """Return a NEW feature dict with the requested overrides applied. Never mutates input."""
    sim = copy.deepcopy(features)

    if "activity_boost_steps_60m" in overrides:
        boost = overrides["activity_boost_steps_60m"]
        sim["steps_roll_sum_60m"] = max(0.0, (sim.get("steps_roll_sum_60m") or 0.0) + boost)
        # a meaningful activity boost also raises recent activity intensity, capped at "vigorous"=3
        intensity_bump = min(1.5, boost / 1000.0)
        sim["activity_intensity_roll_mean_60m"] = min(
            3.0, (sim.get("activity_intensity_roll_mean_60m") or 0.0) + intensity_bump
        )

    if "sleep_adjustment_minutes" in overrides:
        delta = overrides["sleep_adjustment_minutes"]
        sim["sleep_minutes_prev_night"] = max(0.0, (sim.get("sleep_minutes_prev_night") or 0.0) + delta)

    if "glucose_adjustment_mgdl" in overrides:
        delta = overrides["glucose_adjustment_mgdl"]
        for key in ("glucose_mgdl", "glucose_roll_mean_15m", "glucose_roll_mean_30m"):
            if sim.get(key) is not None:
                sim[key] = max(40.0, sim[key] + delta)

    return sim


def run_simulation(baseline_features: dict[str, Any], overrides: dict[str, float]) -> dict[str, Any]:
    """
    Compute baseline vs. simulated prediction/risk for the given overrides.
    `baseline_features` is read, never written to.
    """
    unknown = set(overrides) - set(SIMULATABLE_VARIABLES)
    if unknown:
        raise ValueError(f"Unsupported simulation variable(s): {sorted(unknown)}")

    baseline_pred = predict(baseline_features)
    baseline_risk = assess(baseline_features, baseline_pred["excursion_probability"])

    sim_features = apply_overrides(baseline_features, overrides)
    sim_pred = predict(sim_features)
    sim_risk = assess(sim_features, sim_pred["excursion_probability"])

    return {
        "overrides_applied": overrides,
        "baseline": {"prediction": baseline_pred, "risk": baseline_risk},
        "simulated": {"prediction": sim_pred, "risk": sim_risk},
        "delta_glucose_mgdl": round(
            sim_pred["predicted_glucose_mgdl"] - baseline_pred["predicted_glucose_mgdl"], 1
        ),
        "delta_excursion_probability": round(
            sim_pred["excursion_probability"] - baseline_pred["excursion_probability"], 4
        ),
        "disclaimer": DISCLAIMER,
    }
