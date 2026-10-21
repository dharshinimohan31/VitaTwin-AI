"""
VitaTwin AI — Streamlit doctor dashboard (Phase 9).

Design note on architecture: the dashboard imports the core modules
(digital_twin, models, risk, explainability, simulation) DIRECTLY rather
than calling the FastAPI server over HTTP. This is a deliberate PoC
simplification — `streamlit run` is then the only process needed to see
the whole system work, with no separate server to keep alive. The FastAPI
layer (src/api/main.py) exposes the identical underlying logic for
programmatic/external access and is tested independently. Nothing here is
mocked or hardcoded — every number below comes from the same
preprocessing -> twin -> model -> risk -> explain -> simulate chain the
API uses.

Run: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import settings
from src.digital_twin.recommendations import recommend
from src.digital_twin.risk import assess
from src.digital_twin.state import DigitalTwinEngine
from src.explainability.explain import explain
from src.features.engineering import FEATURE_COLUMNS
from src.models.predict import ModelNotTrainedError, predict
from src.preprocessing.pipeline import run_pipeline
from src.simulation.whatif import SIMULATABLE_VARIABLES, run_simulation

st.set_page_config(page_title="VitaTwin AI", layout="wide", page_icon="🩺")

RISK_COLORS = {"LOW": "#2E7D32", "MODERATE": "#B8860B", "HIGH": "#B3261E"}


@st.cache_data(show_spinner="Loading and preprocessing patient data...")
def load_fused_data() -> pd.DataFrame:
    fused, _ = run_pipeline()
    return fused


@st.cache_resource(show_spinner="Building Digital Twin from patient history...")
def build_engine(patient_id: str, _fused_hash: int) -> DigitalTwinEngine:
    fused = load_fused_data()
    return DigitalTwinEngine.create_from_history(patient_id, fused)


def main():
    st.markdown("## VitaTwin AI")
    st.caption("*Predict. Simulate. Prevent.*")
    st.warning(
        "⚠️ **Research/educational prototype using synthetic data.** Predictions are experimental "
        "model outputs and are NOT medical diagnoses or treatment recommendations.",
        icon="⚠️",
    )

    fused = load_fused_data()
    patient_ids = sorted(fused["patient_id"].unique())

    # ---------------- Panel 1: Patient ----------------
    with st.sidebar:
        st.markdown("### Patient")
        patient_id = st.selectbox("Select patient", patient_ids)

    try:
        engine = build_engine(patient_id, hash(tuple(patient_ids)))
    except Exception as e:
        st.error(f"Could not build twin for {patient_id}: {e}")
        return

    snap = engine.snapshot()
    fv = engine.current_state().features_vector()

    with st.sidebar:
        st.markdown(f"**Patient ID:** {patient_id}")
        st.markdown(f"**Age / Sex:** {snap['demographics']['age']} / {snap['demographics']['sex']}")
        st.markdown(f"**BMI:** {snap['demographics']['bmi']:.1f}")
        st.markdown(f"**HbA1c:** {snap['historical']['hba1c_pct']}%")
        st.markdown(f"**Diagnosis:** {snap['historical']['diagnosis']}")
        st.markdown(f"**Latest observation:** {snap['timestamp']}")

    tabs = st.tabs(["Digital Twin", "Prediction", "Why this prediction?", "What-If Simulator", "Prevent", "History"])

    # ---------------- Panel 2: Digital Twin ----------------
    with tabs[0]:
        st.subheader("Current Twin State")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Glucose", f"{snap['dynamic']['glucose_mgdl']:.0f} mg/dL")
        c2.metric("Heart rate", f"{snap['dynamic']['heart_rate_bpm']:.0f} bpm")
        c3.metric("HRV", f"{snap['dynamic']['hrv_ms']:.0f} ms")
        c4.metric("Steps (interval)", f"{snap['dynamic']['steps']:.0f}")
        c5, c6 = st.columns(2)
        c5.metric("Activity state", snap['dynamic']['activity_state'])
        c6.metric("Asleep now", "Yes" if snap['dynamic']['is_asleep'] else "No")

        dev = fv.get("baseline_glucose_deviation_pct", 0.0)
        st.info(f"Current glucose is **{abs(dev):.1f}% {'above' if dev >= 0 else 'below'}** this patient's own recent baseline (not a population average).")

        hist_df = pd.DataFrame(engine.history(limit=288))  # last ~24h at 5-min resolution
        if not hist_df.empty:
            glucose_series = [h["dynamic"]["glucose_mgdl"] for h in engine.history(limit=288)]
            ts_series = [h["timestamp"] for h in engine.history(limit=288)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_series, y=glucose_series, mode="lines", name="Glucose (observed)", line=dict(color="#1B7F8C")))
            fig.add_hline(y=settings.EXCURSION_THRESHOLD_MGDL, line_dash="dot", line_color="#B3261E",
                          annotation_text="Excursion threshold (180 mg/dL)")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="mg/dL")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- Panel 3: Prediction ----------------
    with tabs[1]:
        st.subheader(f"Prediction — {settings.PREDICTION_HORIZON_MINUTES} minutes ahead")
        try:
            pred = predict(fv)
        except ModelNotTrainedError:
            st.error("No trained model found. Run `python -m src.models.train` first.")
            return
        risk = assess(fv, pred["excursion_probability"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted glucose (+2h)", f"{pred['predicted_glucose_mgdl']:.0f} mg/dL",
                   delta=f"{pred['predicted_glucose_mgdl'] - snap['dynamic']['glucose_mgdl']:.0f} vs. now")
        c2.metric("Excursion probability", f"{pred['excursion_probability']:.0%}")
        color = RISK_COLORS.get(risk["risk_category"], "#444")
        c3.markdown(f"<div style='padding:0.6em;border-radius:8px;background:{color}22;border:1.5px solid {color};'>"
                    f"<b style='color:{color};font-size:1.3em;'>Risk: {risk['risk_category']}</b><br>"
                    f"<span style='color:#444;'>Trend: {risk['trend']}</span></div>", unsafe_allow_html=True)
        st.caption(risk["thresholds_note"])

    # ---------------- Panel 4: Why this prediction ----------------
    with tabs[2]:
        st.subheader("Why this prediction?")
        st.markdown("**Computed contributing factors (risk engine):**")
        for f in risk["contributing_factors"]:
            arrow = "↑" if f["direction"] == "up" else ("↓" if f["direction"] == "down" else "→")
            st.markdown(f"- {arrow} {f['text']}")

        st.markdown("---")
        try:
            exp = explain(fv, FEATURE_COLUMNS)
            st.markdown(f"**Model explanation ({exp['method']}):**")
            for f in exp["top_factors"]:
                arrow = "↑" if f["direction"] == "up" else "↓"
                st.markdown(f"- {arrow} **{f['label']}** (contribution: {f['contribution']:+.3f})")
        except Exception as e:
            st.warning(f"Explainability unavailable: {e}")

    # ---------------- Panel 5: What-If Simulator ----------------
    with tabs[3]:
        st.subheader("What-If Simulator")
        st.caption(
            "Adjust model-supported variables and see how the TRAINED model's prediction changes. "
            "These are real computations on a forked copy of the twin — the real twin is never modified."
        )
        overrides = {}
        for var, spec in SIMULATABLE_VARIABLES.items():
            val = st.slider(spec["label"], spec["min"], spec["max"], spec["default"], spec["step"], key=var)
            if val != spec["default"]:
                overrides[var] = val

        if st.button("Run Simulation", type="primary"):
            if not overrides:
                st.info("Adjust at least one slider above to run a scenario.")
            else:
                try:
                    result = run_simulation(fv, overrides)
                except ModelNotTrainedError:
                    st.error("No trained model found. Run `python -m src.models.train` first.")
                    result = None
                if result:
                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown("#### Baseline")
                        st.metric("Predicted glucose", f"{result['baseline']['prediction']['predicted_glucose_mgdl']:.0f} mg/dL")
                        st.markdown(f"Risk: **{result['baseline']['risk']['risk_category']}** "
                                    f"({result['baseline']['prediction']['excursion_probability']:.0%})")
                    with colB:
                        st.markdown("#### Simulated")
                        st.metric("Predicted glucose", f"{result['simulated']['prediction']['predicted_glucose_mgdl']:.0f} mg/dL",
                                   delta=f"{result['delta_glucose_mgdl']:+.1f} mg/dL")
                        st.markdown(f"Risk: **{result['simulated']['risk']['risk_category']}** "
                                    f"({result['simulated']['prediction']['excursion_probability']:.0%}, "
                                    f"{result['delta_excursion_probability']:+.1%} vs. baseline)")
                    st.warning(result["disclaimer"], icon="⚠️")

    # ---------------- Panel 6: Prevent ----------------
    with tabs[4]:
        st.subheader("Preventive Recommendations")
        try:
            recs = recommend(fv, risk)
            for r in recs:
                badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(r["priority"], "⚪")
                st.markdown(f"{badge} **{r['recommendation']}**")
                st.caption(r["reason"])
        except Exception as e:
            st.warning(f"Recommendations unavailable: {e}")

    # ---------------- History ----------------
    with tabs[5]:
        st.subheader("Recent History (raw observations)")
        recent = engine.history(limit=50)
        if recent:
            table = pd.DataFrame([{
                "timestamp": h["timestamp"], **h["dynamic"]
            } for h in recent])
            st.dataframe(table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
