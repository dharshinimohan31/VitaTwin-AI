# VitaTwin AI — Demo Video Script

**Target length:** 3:30–4:30 (fits the 2–5 min submission window)
**Format:** screen-recorded walkthrough of the live Streamlit dashboard (`streamlit run src/dashboard/app.py`), narrated over it. The system is fully working end to end — this is a real demo, not a mockup.

**Important, verified quirk:** the generated dataset always ends at 23:55 (just before midnight), so every patient's live "current state" in the dashboard defaults to overnight/asleep — which this model correctly scores as LOW risk for all 8 patients. That's not a bug, it's just what time the data happens to stop. The HIGH-risk numbers below (186 mg/dL, 81%) come from a **separate analysis using a truncated history** (VT001's data cut off at 2026-08-13 14:15, an active afternoon moment), run directly against the same trained model — real, but not what you'll see if you just open the dashboard cold. Two honest options for recording:

- **Option A (simplest, still impressive):** demo the live dashboard exactly as it opens — LOW risk, 157 mg/dL — and narrate it as proof the system correctly recognizes a *stable* state too, not just danger. Then use the What-If Simulator's real sliders to show the model respond (e.g. pushing the glucose-proxy slider to +60 moves probability from 0.01% to ~9% — real, modest, honest movement).
- **Option B (more dramatic):** before recording, run the Python snippet below once to confirm current numbers for a truncated-history moment, then narrate those specific numbers while pointing at the architecture/deck slide rather than the live dashboard (the live dashboard has no "jump to a past moment" control yet — that would be a Phase 10 addition).

```python
# Option B helper — run once before recording to get today's real numbers
# for a non-default (daytime) moment, if you want a more dramatic example:
from src.preprocessing.pipeline import run_pipeline
from src.digital_twin.state import DigitalTwinEngine
from src.models.predict import predict
from src.digital_twin.risk import assess
import pandas as pd
fused, _ = run_pipeline()
hist = fused[(fused.patient_id=="VT001") & (fused.timestamp <= pd.Timestamp("2026-08-13 14:15:00"))]
eng = DigitalTwinEngine.create_from_history("VT001", hist)
fv = eng.current_state().features_vector()
pred = predict(fv); risk = assess(fv, pred["excursion_probability"])
print(pred, risk["risk_category"])
```

This script defaults to **Option A** — the real, live, cold-open dashboard state — since it needs no setup and is equally honest.

---

## [0:00–0:20] Hook + Problem

**Visual:** Title slide (from `presentation_deck.pptx`), then Problem slide.

**Narration:**
> "Healthcare today is reactive — a patient's data and wearable signals sit unused until something goes wrong. We built VitaTwin AI, a working Digital Twin prototype, to change that."

## [0:20–0:40] The Idea

**Visual:** Paradigm-shift slide.

**Narration:**
> "VitaTwin AI fuses a patient's medical history with live wearable and glucose data into one continuously-updated Digital Twin, then predicts, simulates, and recommends — not just dashboards."

## [0:40–1:15] Meet the Twin

**Visual:** Dashboard, Digital Twin tab. Select patient VT001.

**Narration:**
> "Here's patient VT001 — Type 2 Diabetes, HbA1c 9.2%. On the left, their static record. Here in the Digital Twin tab, their live state: current glucose, heart rate, HRV, activity, sleep — and their glucose trend over the last 24 hours, plotted against the excursion threshold."

## [1:15–1:50] Prediction — real model output

**Visual:** Prediction tab.

**Narration (real numbers — confirm against what's actually on screen when you record):**
> "The system predicts glucose two hours ahead using a Random Forest trained on this patient population — test-set mean absolute error of 11 milligrams per deciliter, ROC-AUC of 0.92 for flagging excursions. Right now, for this patient, it's predicting 157 milligrams per deciliter — a very low probability of crossing the hyperglycemic threshold. Risk: LOW. That's genuinely good — it means the model isn't crying wolf at a quiet, overnight moment."

## [1:50–2:20] Why this prediction?

**Visual:** "Why this prediction?" tab.

**Narration:**
> "We never show a bare risk score, high or low. Here's the risk engine's own computed factors for this exact moment — glucose trend, deviation from this patient's personal baseline, recent activity, recent sleep — and below that, real SHAP values from the trained model itself, ranking exactly which features shaped this prediction."

## [NEW, recommended] Live Twin — the strongest moment to show

**Visual:** Live Twin tab. Press Play at 4x or 12x speed.

**Narration:**
> "Instead of a static snapshot, you can watch this patient's whole day replay — the Digital Twin updates every 5 minutes, and the model re-predicts and re-assesses risk live, right here. Watch the risk ring move from LOW overnight, up through MODERATE and HIGH during the day, and back down — this isn't an animation, it's the real trained model running 288 times on real historical data."

Pause on a HIGH moment (e.g. mid-morning) before continuing to "Why this prediction?" — gives you a dramatic, 100% real HIGH-risk state to narrate, solving the "default state is LOW" issue from Option A/B above without needing any workaround.

## [2:20–3:10] What-If Simulator — real, honest result

**Visual:** What-If Simulator tab. Move the glucose-proxy slider to its maximum (+60). Click "Run Simulation."

**Narration (real numbers, Option A):**
> "This is where it gets useful. I'll simulate a hypothetical rise in current glucose for this same patient — the system reruns the SAME trained model on a forked copy of the state, never touching the real twin. Result: predicted glucose moves from 157 to about 166, and excursion probability moves from near-zero to about 9%. Still LOW — and that's honest: this model responds proportionally, it doesn't overreact to a single hypothetical input. The full range of what it does under larger shifts is documented plainly in our model notes, limitations included."

## [3:10–3:40] Prevent

**Visual:** Prevent tab.

**Narration:**
> "Based on this same computed state, the system generates specific, conditional recommendations — not a static list. Low activity, short sleep, and the HIGH risk level each independently trigger here, and a HIGH risk always includes a prompt to consult a professional, never a diagnosis."

## [3:40–4:10] Architecture + honesty

**Visual:** Architecture diagram slide.

**Narration:**
> "Under the hood: synthetic, disclosed data, a real preprocessing pipeline, a stateful Digital Twin engine, trained models with temporally-validated metrics, SHAP explainability, and a forked-state simulator — all served through FastAPI and this dashboard. Every number you just saw came from running this code, not a mockup."

## [4:10–4:30] Close

**Visual:** Safety/closing slide.

**Narration:**
> "VitaTwin AI is a research prototype — not a diagnosis. But it's a real, working demonstration of proactive, explainable, simulation-driven healthcare. Thank you."

---

## Recording Checklist
- [ ] Run `python -m src.data.generate_data`, `python -m src.models.train`, then `streamlit run src/dashboard/app.py` BEFORE recording — confirm the dashboard loads with no errors
- [ ] Screen resolution ≥1080p, browser zoomed so tab content is legible
- [ ] Do one full silent dry-run first and confirm what risk category VT001 (or your chosen patient) actually shows — it will be LOW by default (data always ends at night) unless you regenerate with different parameters or follow Option B above
- [ ] Narrate the real numbers shown on screen, not the ones in this script, if they differ — these are the result of one specific run (seed 42) and will match a fresh `generate_data.py --seed 42` run exactly, but always say what's actually on screen
- [ ] Export as MP4, upload **unlisted** to YouTube, or use OBS/screen-record + any video editor for trimming
