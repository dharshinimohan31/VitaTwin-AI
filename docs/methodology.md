# VitaTwin AI — Methodology

## Pipeline

```
Synthetic generation (src/data/generate_data.py)
        ↓
Preprocessing & fusion (src/preprocessing/pipeline.py)
  — schema validation, duplicate/out-of-order handling, bounds clipping,
    short-gap interpolation, static+dynamic join
        ↓
Feature engineering (src/features/engineering.py)
  — rolling windows, personal baseline, circadian encoding, prediction targets
        ↓
Digital Twin (src/digital_twin/state.py)
  — VitaTwinState: create / update / snapshot / history / fork
        ↓
Prediction (src/models/predict.py, trained by src/models/train.py)
  — RandomForest regression (glucose @ +2h) + classification (excursion risk)
        ↓
Risk engine (src/digital_twin/risk.py)
  — documented thresholds + computed contributing factors
        ↓
Explainability (src/explainability/explain.py)
  — SHAP TreeExplainer, per-prediction
        ↓
What-If simulation (src/simulation/whatif.py)
  — forked feature vector, re-scored by the SAME trained model
        ↓
Recommendations (src/digital_twin/recommendations.py)
  — condition-triggered, not static per patient
        ↓
FastAPI (src/api/main.py)  +  Streamlit dashboard (src/dashboard/app.py)
```

## Digital Twin Design

`VitaTwinState` is a real dataclass, not a dict assembled for display. `DigitalTwinEngine`
owns one patient's live state and an append-only history log. `engine.fork()`
deep-copies both state and history for simulation — the forked copy can be mutated
freely with zero effect on the original (enforced by
`tests/test_digital_twin.py::test_fork_does_not_mutate_original_state` and
`tests/test_simulation.py::test_run_simulation_never_mutates_real_twin_state`).

## Validation Discipline

Temporal, per-patient split (70/15/15, chronological) — see `docs/model.md` for why a
random split would leak. Every model-selection decision (Ridge vs. Random Forest,
Logistic Regression vs. Random Forest) is made on the validation split; the test split
is touched exactly once, for final reported metrics.

## Why RandomForest, not a sequence model

Phase 0 planned to start with simple baselines (Logistic Regression → Random Forest →
gradient boosting) and only reach for LSTM/GRU if the data genuinely justified it. The
engineered rolling-window features already capture the short-term temporal structure a
sequence model would otherwise have to learn from raw sequences, and with only 8
patients × 14 days, a sequence model would be far more likely to overfit than to add
real signal. Random Forest beat the linear baseline on both tasks (regression val MAE
10.7 vs. 15.9; classification val ROC-AUC 0.917 vs. 0.848) — see `docs/model.md`.

## Reproducibility

`GLOBAL_SEED = 42` (src/config/settings.py) seeds synthetic data generation and every
model's `random_state`. Running `generate_data.py` → `train.py` end to end from a clean
checkout reproduces the same metrics reported in `docs/model.md`.
