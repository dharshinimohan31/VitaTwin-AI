# VitaTwin AI — Model Documentation

All numbers on this page are copied directly from `models/metrics.json`, produced by
running `python -m src.models.train` against the generated synthetic dataset
(8 patients × 14 days × 5-min resolution). Nothing here is invented or illustrative.

## Prediction Target

- **Regression:** glucose value 120 minutes (2 hours) ahead, in mg/dL (`target_glucose_future`)
- **Classification:** probability that glucose will be ≥ 180 mg/dL at that same 2-hour
  horizon (`target_excursion`). 180 mg/dL is a commonly-used general hyperglycemia
  reference point, used here only to define a classification task on synthetic
  data — **not a diagnostic threshold**.

## Features

22 features — see `docs/features.md` for the full dictionary. Rolling-window glucose/HR/HRV/activity
statistics, a personal (expanding, per-patient) glucose baseline, circadian time-of-day
encoding, and static demographic/EHR context.

## Models Evaluated

| Target | Candidates | Selection metric |
|---|---|---|
| Regression | Ridge Regression, Random Forest Regressor | Validation MAE |
| Classification | Logistic Regression, Random Forest Classifier | Validation ROC-AUC |

Both candidate sets were evaluated on the SAME validation split; the better validation
score wins and is the one reported on the held-out test set below. Random Forest won
both — reasonable given the engineered features are a mix of smooth rolling stats and
categorical-ish activity/sleep signals, which tree ensembles handle well, and matches
the Phase 0 plan to prefer gradient-boosted/tree baselines over immediately reaching
for a sequence model.

## Validation Strategy — Temporal, No Leakage

Each patient's timeline is split chronologically: **first 70% = train, next 15% =
validation, final 15% = test** — per patient, then concatenated. This is NOT a random
row split. Adjacent 5-minute readings are highly autocorrelated; a random split would
let the model see glucose readings from minutes before/after a "held out" point during
training, silently inflating every metric. `tests/test_models.py` asserts the split
stays chronological and patient-separated.

## Test-Set Results (actual, from `models/metrics.json`)

**Regression — glucose @ +2h:**

| Metric | Value |
|---|---|
| MAE | **11.0 mg/dL** |
| RMSE | 17.0 mg/dL |
| R² | 0.657 |
| Test rows | 4,648 |

**Classification — excursion (≥180 mg/dL) @ +2h:**

| Metric | Value |
|---|---|
| Accuracy | 83.4% |
| Precision | 49.1% |
| Recall | 84.0% |
| F1 | 0.620 |
| ROC-AUC | **0.919** |
| PR-AUC | 0.677 |
| Positive rate (test set) | 16.2% |

**Reading these honestly:** ROC-AUC 0.92 is strong separation for a first baseline.
Precision (49%) is modest relative to recall (84%) — the model is tuned toward catching
excursions (high recall) at the cost of some false alarms, which is the right
trade-off direction for an early-warning tool, but worth stating plainly rather than
only citing the headline ROC-AUC.

**Per-patient generalization** (regression MAE on each patient's own held-out test
window): ranges from 9.4 to 12.9 mg/dL across the 8 patients — no patient is a wild
outlier, but with only 8 synthetic patients this is a weak signal about real
generalization, not a strong one. See Limitations.

## Risk Thresholds

```
LOW:      excursion_probability < 33%
MODERATE: 33% <= excursion_probability < 66%
HIGH:     excursion_probability >= 66%
```

A simple documented tertile-style split (`src/config/settings.py`), chosen so all three
categories are reachable on this dataset — not derived from a clinical scoring system,
and not claimed to be. See `docs/safety.md`.

## Explainability

SHAP `TreeExplainer` against the trained Random Forest classifier (`src/explainability/explain.py`),
computed per-prediction. If SHAP is unavailable in a given environment, the code falls
back to signed global `feature_importances_` and labels that output
`"feature_importance_fallback"` rather than presenting it as SHAP.

## Known Limitations (stated plainly)

1. **Trained on synthetic, physiologically-simulated data** — not real patients. See
   `docs/dataset.md` for full data-provenance disclosure.
2. **What-If simulation sensitivity is modest.** Across multiple real test runs
   (`docs/demo_script.md`), activity/sleep lever changes shift the predicted excursion
   probability by single-digit-to-low-double-digit percentage points when starting
   glucose is already high — the model is (correctly, for this data) dominated by
   *current and recent glucose* far more than by short-horizon lifestyle levers alone.
   A richer feature set (meal/carbohydrate intake, medication timing) would likely let
   the simulator show larger, more clinically intuitive swings — flagged as future
   work, not hidden.
3. **Only 8 synthetic patients.** Per-patient test MAE looks consistent, but this is
   nowhere near enough patients to claim real generalization.
4. **No hyperparameter tuning performed** — both models used reasonable defaults, not a
   tuned search. A straightforward next step, intentionally deferred to keep this
   baseline honest about what it is.
5. **Not clinically validated. Not a diagnostic tool.** See `docs/safety.md`.
