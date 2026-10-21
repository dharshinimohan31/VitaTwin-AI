# VitaTwin AI
### *Predict. Simulate. Prevent.*

Submission for the **Happiest Health Digital Twin Challenge 2026**

> ⚠️ **This is a research/educational prototype.** It is not a medical
> diagnosis system, does not replace doctors, and does not provide
> guaranteed treatment recommendations. All predictions and simulations are
> experimental model outputs. See [`docs/safety.md`](docs/safety.md).

**Status: working end-to-end prototype.** Data generation → preprocessing →
Digital Twin → trained ML models → risk engine → SHAP explainability →
What-If simulation → preventive recommendations → FastAPI → Streamlit
dashboard are all implemented and tested (**66/66 tests passing**). The one
remaining item is recording the demo video against the finished system —
see Project Status below.

---

## Team & College

| Field | Value |
|---|---|
| Team Name | A Girl :) |
| College / Incubator | SRM Madurai College for Engineering and Technology |
| Team Leader | Dharshini M |
| Team Members | `<LIST MEMBERS, IF ANY>` |

*(Repo/folder name per the submission rules: `A Girl :)_SRM Madurai College for Engineering and Technology` — Windows/GitHub may reject the `:` and `)` characters in a folder/repo name; consider a safe variant like `AGirl_SRMMaduraiCET` if you hit errors creating it.)*

## Project Title

**VitaTwin AI** — a working AI-powered healthcare Digital Twin proof-of-concept for early glycemic-risk awareness in Type 2 Diabetes.

## Problem Statement & Healthcare Use Case

Today's healthcare model is reactive: **DATA → EVENT → REACTION**. Type 2 Diabetes is
one of India's most prevalent chronic, lifestyle-linked conditions, and unpredicted
hyperglycemic excursions are a leading driver of avoidable complications. VitaTwin AI
reframes this as **DATA → DIGITAL TWIN → PREDICTION → SIMULATION → EARLY AWARENESS**: a
continuously-updated per-patient Digital Twin, a trained model that forecasts glucose
2 hours ahead, and a What-If simulator a clinician can actually interact with.

## Solution Summary — What Actually Works

- **Digital Twin engine** (`src/digital_twin/state.py`): a real `VitaTwinState` object
  per patient — not a dict assembled for display — with create/update/snapshot/history/fork.
- **Trained prediction models** (`src/models/`): Random Forest regression (glucose @ +2h,
  **test MAE 11.0 mg/dL**) and classification (excursion risk, **test ROC-AUC 0.92**),
  selected over linear baselines on a validation split, evaluated once on a held-out
  temporal test split. Full numbers: [`docs/model.md`](docs/model.md).
- **Risk engine** (`src/digital_twin/risk.py`): documented probability thresholds +
  computed contributing factors — never a bare score.
- **Explainability** (`src/explainability/explain.py`): real SHAP `TreeExplainer` output
  per prediction, with a labeled fallback if SHAP is unavailable.
- **What-If simulator** (`src/simulation/whatif.py`): forks the twin's feature vector,
  re-runs the *same trained model*, and is proven never to mutate the real twin state
  (tested explicitly). See `docs/model.md` for an honest discussion of what it does and
  doesn't show.
- **Preventive recommendations** (`src/digital_twin/recommendations.py`): condition-triggered,
  verified to differ across different patient states — not a static list.
- **FastAPI backend** (`src/api/main.py`): 12 real endpoints (including Live Twin replay), all computing from the
  modules above, no hardcoded responses.
- **Two working dashboards**, both wired to real computation, no mock data:
  - `src/dashboard/app.py` — Streamlit, 6 panels, calls the core modules directly.
  - `src/dashboard/web/index.html` — a custom dark/glass UI (patient rail, live risk
    ring, real SHAP factors, working What-If sliders) that calls the FastAPI backend
    over HTTP via `fetch()`. Built because the brief asked for something beyond a
    generic data-app look; screenshot-tested end-to-end with Playwright against the
    real running API (not a static mockup).

## Technical Stack

| Layer | Technology |
|---|---|
| Data generation | Python, NumPy, Pandas — rule-based physiological simulator |
| Preprocessing / features | Pandas — temporal, leakage-free, patient-isolated |
| ML models | scikit-learn: Ridge/RandomForestRegressor, LogisticRegression/RandomForestClassifier |
| Explainability | SHAP (TreeExplainer) |
| Backend API | FastAPI + Pydantic |
| Database | SQLite (`src/data/db.py`) |
| Dashboard | Streamlit + Plotly |
| Testing | pytest (66 tests) |

## Data Sources & Provenance

Only synthetic, anonymized, or open datasets are used, per challenge rules. **Full
disclosure in [`docs/dataset.md`](docs/dataset.md):** static EHR is synthetically
generated for a realistic T2D-relevant population; the dynamic (glucose/HR/HRV/steps/
sleep) stream is a documented rule-based physiological simulator, schema-compatible
with real datasets (OhioT1DM/D1NAMO) for future substitution. Every row is tagged
`data_source`.

## Architecture

[`docs/submission/architecture_diagram.pptx`](docs/submission/architecture_diagram.pptx) /
[`.pdf`](docs/submission/architecture_diagram.pdf) · written walkthrough:
[`docs/architecture.md`](docs/methodology.md) (see `docs/methodology.md` for the
implemented pipeline, as-built).

## Demo Video

`<INSERT 2–5 MIN VIDEO LINK — unlisted YouTube or narrated recording>`

Script with real, verified numbers: [`docs/demo_script.md`](docs/demo_script.md).
The dashboard is fully functional — this just needs to be recorded.

## Presentation

[`docs/submission/presentation_deck.pptx`](docs/submission/presentation_deck.pptx) /
[`.pdf`](docs/submission/presentation_deck.pdf) — including an honest What-If example
and an accurate progress slide (no illustrative/fabricated numbers).

## License

[MIT License](LICENSE) — covers source code and documentation only; see the note in
`LICENSE` on medical/regulatory scope.

## Installation

```bash
git clone <REPO_URL>
cd AGirl_SRMMaduraiCET   # or whatever safe variant you name the repo folder
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage — Full Pipeline, In Order

```bash
# 1. Generate the synthetic dataset (8 patients x 14 days x 5-min resolution)
python -m src.data.generate_data --n-patients 8 --n-days 14 --seed 42

# 2. (Optional) sanity-check the preprocessing pipeline standalone
python -m src.preprocessing.pipeline

# 3. Train the models — prints real metrics, saves models/*.joblib + models/metrics.json
python -m src.models.train

# 4. (Optional) initialize/seed the SQLite database
python -m src.data.db

# 5. Start the API
uvicorn src.api.main:app --reload --port 8000
# then see http://127.0.0.1:8000/docs for interactive Swagger UI

# 6. Start a dashboard — two options, both real, both call the same backend logic:

#    Option A: Streamlit (functional, fast to run, default data-app look)
#    does not require the API running — uses the core modules directly
streamlit run src/dashboard/app.py

#    Option B: custom web dashboard (premium dark UI) — REQUIRES the API
#    running (step 5) since it's a static page that calls it over HTTP
uvicorn src.api.main:app --port 8000          # terminal 1
python -m http.server 8080 --directory src/dashboard/web   # terminal 2
# then open http://127.0.0.1:8080/index.html
```

## Model Training & Results

`python -m src.models.train` — trains Ridge + Random Forest (regression) and Logistic
Regression + Random Forest (classification) on a per-patient chronological 70/15/15
split, selects the better validation performer for each task, evaluates once on the
held-out test split, and saves `models/metrics.json`. **Real results:** regression test
MAE 11.0 mg/dL (R²=0.657); classification test ROC-AUC 0.919, F1 0.620. Full
breakdown, including per-patient generalization and stated limitations:
[`docs/model.md`](docs/model.md).

## API

12 endpoints — `/health`, `/patients`, `/patients/{id}`, `/patients/{id}/twin`,
`/patients/{id}/history`, `/patients/{id}/prediction`, `/patients/{id}/risk`,
`/patients/{id}/explain`, `/patients/{id}/recommendations`, `/patients/{id}/replay-meta`,
`/patients/{id}/replay-frames`, `/patients/{id}/replay/{step}`, and
`POST /patients/{id}/simulate`. Interactive docs at `/docs` once running. All tested
in `tests/test_api.py`.

## Testing

```bash
pytest tests/ -v
```

**66/66 passing** — data generation (11), preprocessing (7, including deliberately
corrupted inputs: duplicate/out-of-order/missing/out-of-bounds data), feature
engineering (5, including an explicit no-leakage check), Digital Twin (8, including
fork-isolation safety), risk/recommendations (7), What-If simulation (7, including the
critical "never mutates the real twin" safety assertion), models (6), API (9).

## Project Status / Roadmap

- [x] Phase 0 — Feasibility report
- [x] Phase 1 — Synthetic data generator
- [x] Phase 2 — Preprocessing & fusion pipeline
- [x] Phase 3 — Digital Twin engine
- [x] Phase 4-5 — Trained models, real evaluated metrics
- [x] Phase 6 — SHAP explainability
- [x] Phase 7 — What-If simulator + preventive recommendations
- [x] Phase 8 — FastAPI backend
- [x] Phase 9 — Streamlit dashboard
- [x] Phase 10 — Live Twin replay: a "Live Twin" tab replays each patient's last 24h, re-running the real trained model at every step (Play/Pause/Reset/Scrub, variable speed), backed by two new tested API endpoints (`/replay-frames`, `/replay/{step}`)
- [x] Phase 11 — Full test suite (66/66)
- [x] Phase 12 — Documentation
- [ ] Phase 13 — Record the actual demo video (script ready, numbers verified against running code)

## Limitations

See [`docs/model.md`](docs/model.md) §Limitations for the full, honest list — notably:
trained on synthetic data; What-If simulation shows modest effect sizes because the
baseline model is (correctly) dominated by current glucose over short-horizon lifestyle
levers; only 8 synthetic patients; no hyperparameter tuning performed yet.

## Safety

See [`docs/safety.md`](docs/safety.md) — displayed on every dashboard screen that shows
a prediction, and in the API's `/health` response.

## Future Work

Real CGM+wearable dataset integration
(OhioT1DM/D1NAMO); richer meal/medication features to sharpen What-If sensitivity;
hyperparameter tuning; multi-condition expansion (hypertension, cardiovascular risk).
