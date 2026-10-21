# VitaTwin AI — Feature Dictionary

All features are computed in `src/features/engineering.py`. Every rolling/derived
feature at row *t* uses only data at or before timestamp *t* (verified by
`tests/test_features.py::test_rolling_features_use_only_past_data_no_leakage`) — this is
what makes the temporal train/val/test split in `src/models/train.py` leakage-free.

| Feature | Meaning | Calculation | Source | Raw / Derived |
|---|---|---|---|---|
| `glucose_mgdl` | Current glucose reading | — | sensor stream | Raw |
| `glucose_roll_mean_15m` / `_30m` / `_60m` / `_120m` | Rolling average glucose over the stated window | Pandas rolling mean, `min_periods` = 1/3 of window | derived from glucose_mgdl | Derived |
| `glucose_slope_15m` | Short-term glucose trend | `(glucose_now − glucose_15min_ago) / 15` (mg/dL per minute) | derived | Derived |
| `glucose_roll_std_60m` | Glucose variability | Rolling std-dev, 60-min window | derived | Derived |
| `heart_rate_bpm` | Current heart rate | — | sensor stream | Raw |
| `hr_roll_mean_30m` | 30-min average heart rate | Rolling mean | derived | Derived |
| `hrv_ms` | Current HRV (rMSSD-like proxy) | — | sensor stream | Raw |
| `hrv_roll_mean_60m` | 60-min average HRV | Rolling mean | derived | Derived |
| `steps_roll_sum_60m` | Steps in the last 60 minutes | Rolling sum | derived from steps | Derived |
| `activity_intensity_roll_mean_60m` | Recent activity level (0=sleep/sedentary … 3=vigorous) | `activity_state` mapped to ordinal 0-3, rolling mean | derived | Derived |
| `sleep_minutes_prev_night` | Minutes asleep during the most recent overnight block (00:00-09:00) | Sum of `is_asleep` minutes in that window, assigned to the following day's rows | derived | Derived |
| `is_asleep_now` | Whether the patient is asleep at this instant | `is_asleep` cast to int | sensor stream | Raw (recast) |
| `hour_sin` / `hour_cos` | Circadian time-of-day encoding | `sin`/`cos` of `2π × hour_fraction / 24` | derived from timestamp | Derived |
| `baseline_glucose_deviation_pct` | % deviation from this patient's OWN glucose history | `(glucose_now − expanding_mean_glucose_so_far) / expanding_mean × 100`, computed with a 1-row shift so a reading never includes itself | derived (per-patient, expanding) | Derived |
| `age`, `bmi`, `hba1c_pct`, `diabetes_duration_years` | Static patient context | — | synthetic EHR | Raw |

## Prediction targets (`add_prediction_target`)

| Target | Definition |
|---|---|
| `target_glucose_future` | `glucose_mgdl` shifted `PREDICTION_HORIZON_STEPS` (24 steps = 2h) into the future |
| `target_excursion` | `1` if `target_glucose_future >= EXCURSION_THRESHOLD_MGDL` (180 mg/dL), else `0` |

Rows within the final 2 hours of each patient's timeline have no future value to predict
and are dropped before training (NaN target).

## Deliberately NOT included

- Direct meal/carbohydrate intake as a feature — the synthetic generator's meal events
  are a *source* of glucose dynamics, not a feature exposed to the model (the model has
  to learn from glucose's own trajectory, like a real deployment without meal-logging
  would). This is a known limitation — see `docs/model.md` §Limitations.
- Any field derived from the TARGET window itself (no "future glucose" leaking back as
  a predictor under a different name).
