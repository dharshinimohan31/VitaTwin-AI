"""
VitaTwin AI — risk engine (Phase 3/6).

Converts a model's excursion_probability into a documented risk category
and a short list of signed contributing factors a clinician can read
without needing to understand the underlying model. Thresholds are fixed,
documented constants (src/config/settings.py) — not tuned per-patient, not
invented at call time.
"""

from __future__ import annotations

from typing import Any

from src.config import settings


def categorize(excursion_probability: float) -> str:
    if excursion_probability < settings.RISK_LOW_MAX_PROB:
        return "LOW"
    if excursion_probability < settings.RISK_MODERATE_MAX_PROB:
        return "MODERATE"
    return "HIGH"


def trend_direction(current_glucose: float, roll_mean_60m: float) -> str:
    if current_glucose > roll_mean_60m * 1.03:
        return "rising"
    if current_glucose < roll_mean_60m * 0.97:
        return "falling"
    return "stable"


def assess(twin_features: dict[str, Any], excursion_probability: float) -> dict[str, Any]:
    """
    Build the full risk assessment the dashboard/API return — category,
    trend, and the specific computed factors that justify it. Every factor
    below reads directly from the twin's actual derived features; nothing
    here is templated per risk-category.
    """
    category = categorize(excursion_probability)
    trend = trend_direction(
        twin_features.get("glucose_mgdl", 0.0),
        twin_features.get("glucose_roll_mean_60m", twin_features.get("glucose_mgdl", 0.0)),
    )

    factors = []
    slope = twin_features.get("glucose_slope_15m", 0.0) or 0.0
    if slope > 0.15:
        factors.append({"direction": "up", "text": f"Glucose rising ~{slope:.2f} mg/dL/min over the last 15 min"})
    elif slope < -0.15:
        factors.append({"direction": "down", "text": f"Glucose falling ~{abs(slope):.2f} mg/dL/min over the last 15 min"})

    dev = twin_features.get("baseline_glucose_deviation_pct", 0.0) or 0.0
    if abs(dev) > 10:
        d = "up" if dev > 0 else "down"
        factors.append({"direction": d, "text": f"{abs(dev):.1f}% {'above' if dev > 0 else 'below'} this patient's personal glucose baseline"})

    activity = twin_features.get("activity_intensity_roll_mean_60m", 0.0) or 0.0
    if activity < 0.3:
        factors.append({"direction": "down", "text": "Low recent activity (last 60 min mostly sedentary/asleep)"})

    sleep_min = twin_features.get("sleep_minutes_prev_night")
    if sleep_min is not None and sleep_min < 360:
        factors.append({"direction": "down", "text": f"Short sleep last night (~{sleep_min:.0f} min, under 6h)"})

    if not factors:
        factors.append({"direction": "flat", "text": "No single dominant factor — state is near this patient's typical pattern"})

    return {
        "risk_category": category,
        "excursion_probability": round(excursion_probability, 4),
        "trend": trend,
        "contributing_factors": factors,
        "thresholds_note": (
            f"LOW < {settings.RISK_LOW_MAX_PROB:.0%} <= MODERATE < "
            f"{settings.RISK_MODERATE_MAX_PROB:.0%} <= HIGH (of predicted excursion probability). "
            "Fixed, documented thresholds — see docs/model.md. Not a clinical diagnosis."
        ),
    }
