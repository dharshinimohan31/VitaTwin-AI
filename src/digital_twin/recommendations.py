"""
VitaTwin AI — preventive recommendation engine (Phase 7/satisfies §15).

Every recommendation is conditional on the twin's ACTUAL computed
features/risk for THIS patient at THIS moment — nothing here fires
unconditionally, so two patients (or the same patient at two different
times) can and will get different recommendation sets. This is decision
support, not treatment: every item in the catalogue is framed as
"consider" / "suggests", never an instruction, and a HIGH-risk result
always includes a professional-consultation nudge rather than a specific
clinical action.
"""

from __future__ import annotations

from typing import Any


def recommend(twin_features: dict[str, Any], risk: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a priority-ordered list of {trigger, recommendation, reason, priority}."""
    items: list[dict[str, Any]] = []

    activity = twin_features.get("activity_intensity_roll_mean_60m", 0.0) or 0.0
    if activity < 0.3:
        items.append({
            "trigger": "low_recent_activity",
            "recommendation": "Consider a short walk or light activity in the next hour.",
            "reason": f"Recent activity level ({activity:.2f}) is at or near sedentary for the last 60 minutes.",
            "priority": "medium",
        })

    sleep_min = twin_features.get("sleep_minutes_prev_night")
    if sleep_min is not None and sleep_min < 360:
        items.append({
            "trigger": "short_sleep",
            "recommendation": "Prioritize earlier bedtime tonight — recent sleep has been short.",
            "reason": f"Last night's sleep (~{sleep_min:.0f} min) was under the 6-hour reference used here.",
            "priority": "medium",
        })

    dev = twin_features.get("baseline_glucose_deviation_pct", 0.0) or 0.0
    if dev > 15:
        items.append({
            "trigger": "elevated_vs_personal_baseline",
            "recommendation": "Current glucose is notably above this patient's own recent pattern — worth noting at next check-in.",
            "reason": f"Glucose is {dev:.1f}% above the patient's personal rolling baseline, not just a population average.",
            "priority": "medium",
        })

    slope = twin_features.get("glucose_slope_15m", 0.0) or 0.0
    if slope > 0.3:
        items.append({
            "trigger": "rapid_glucose_rise",
            "recommendation": "Glucose is rising quickly — re-check in 15-30 minutes.",
            "reason": f"Glucose rising at ~{slope:.2f} mg/dL/min over the last 15 minutes.",
            "priority": "high",
        })

    category = risk.get("risk_category")
    if category == "HIGH":
        items.append({
            "trigger": "high_predicted_risk",
            "recommendation": "Predicted risk is HIGH — this is a prompt to involve a healthcare professional, not a diagnosis.",
            "reason": f"Model-predicted excursion probability is {risk.get('excursion_probability', 0):.0%}, above the HIGH threshold.",
            "priority": "high",
        })

    if not items:
        items.append({
            "trigger": "stable_state",
            "recommendation": "No specific action indicated — current state is within this patient's typical pattern.",
            "reason": "No monitored factor crossed its threshold.",
            "priority": "low",
        })

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(items, key=lambda x: order.get(x["priority"], 3))
