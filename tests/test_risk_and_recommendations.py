from src.config import settings
from src.digital_twin.recommendations import recommend
from src.digital_twin.risk import assess, categorize, trend_direction


def test_categorize_boundaries():
    assert categorize(0.0) == "LOW"
    assert categorize(settings.RISK_LOW_MAX_PROB - 0.01) == "LOW"
    assert categorize(settings.RISK_LOW_MAX_PROB) == "MODERATE"
    assert categorize(settings.RISK_MODERATE_MAX_PROB) == "HIGH"
    assert categorize(1.0) == "HIGH"


def test_trend_direction():
    assert trend_direction(130, 100) == "rising"
    assert trend_direction(90, 100) == "falling"
    assert trend_direction(100, 100) == "stable"


def test_assess_flags_rapid_rise_and_baseline_deviation():
    features = {
        "glucose_mgdl": 180, "glucose_roll_mean_60m": 150,
        "glucose_slope_15m": 0.5, "baseline_glucose_deviation_pct": 20,
        "activity_intensity_roll_mean_60m": 0.1, "sleep_minutes_prev_night": 300,
    }
    risk = assess(features, 0.8)
    assert risk["risk_category"] == "HIGH"
    texts = " ".join(f["text"] for f in risk["contributing_factors"])
    assert "rising" in texts
    assert "baseline" in texts
    assert "activity" in texts.lower()
    assert "sleep" in texts.lower()


def test_assess_reports_no_dominant_factor_when_nothing_crosses_threshold():
    features = {
        "glucose_mgdl": 100, "glucose_roll_mean_60m": 100,
        "glucose_slope_15m": 0.0, "baseline_glucose_deviation_pct": 1.0,
        "activity_intensity_roll_mean_60m": 1.5, "sleep_minutes_prev_night": 420,
    }
    risk = assess(features, 0.05)
    assert risk["risk_category"] == "LOW"
    assert len(risk["contributing_factors"]) == 1
    assert "typical pattern" in risk["contributing_factors"][0]["text"]


def test_recommend_differs_for_different_states():
    low_activity_state = {"activity_intensity_roll_mean_60m": 0.0, "sleep_minutes_prev_night": 420,
                           "baseline_glucose_deviation_pct": 0, "glucose_slope_15m": 0.0}
    normal_state = {"activity_intensity_roll_mean_60m": 2.0, "sleep_minutes_prev_night": 420,
                     "baseline_glucose_deviation_pct": 0, "glucose_slope_15m": 0.0}
    risk_low = {"risk_category": "LOW", "excursion_probability": 0.05}

    recs_a = recommend(low_activity_state, risk_low)
    recs_b = recommend(normal_state, risk_low)

    triggers_a = {r["trigger"] for r in recs_a}
    triggers_b = {r["trigger"] for r in recs_b}
    assert "low_recent_activity" in triggers_a
    assert "low_recent_activity" not in triggers_b
    assert triggers_a != triggers_b  # proves recommendations are NOT a static per-patient list


def test_recommend_high_risk_includes_professional_consultation_nudge():
    state = {"activity_intensity_roll_mean_60m": 2.0, "sleep_minutes_prev_night": 420,
              "baseline_glucose_deviation_pct": 0, "glucose_slope_15m": 0.0}
    risk_high = {"risk_category": "HIGH", "excursion_probability": 0.85}
    recs = recommend(state, risk_high)
    assert any(r["trigger"] == "high_predicted_risk" for r in recs)
    high_rec = next(r for r in recs if r["trigger"] == "high_predicted_risk")
    assert "not a diagnosis" in high_rec["recommendation"]
    assert recs[0]["priority"] == "high"  # high-priority items sorted first


def test_recommend_never_empty():
    neutral_state = {"activity_intensity_roll_mean_60m": 1.5, "sleep_minutes_prev_night": 420,
                      "baseline_glucose_deviation_pct": 0, "glucose_slope_15m": 0.0}
    risk_low = {"risk_category": "LOW", "excursion_probability": 0.01}
    recs = recommend(neutral_state, risk_low)
    assert len(recs) >= 1
