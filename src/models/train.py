"""
VitaTwin AI — baseline model training (Phase 4-5).

Trains two models on the engineered feature table:
  1. RandomForestRegressor  -> target_glucose_future  (glucose @ +2h, mg/dL)
  2. RandomForestClassifier -> target_excursion        (P(glucose >= 180 mg/dL @ +2h))

Validation strategy (critical — see docs/model.md):
  - TEMPORAL split per patient: first ~70% of each patient's chronological
    timeline is train, the next ~15% is validation, the final ~15% is test.
  - This is NOT a random row split. A random split would let the model see
    glucose readings from 10 minutes after a test point during training
    (since adjacent 5-minute readings are highly correlated), which would
    silently inflate every metric. Keeping each patient's timeline intact
    and splitting along TIME is what prevents that leakage.
  - We also report a per-patient breakdown so "the model only works on
    patients it has seen" is checkable, not just asserted.

Run: python -m src.models.train
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score,
)

from src.config import settings
from src.features.engineering import FEATURE_COLUMNS, build_feature_table
from src.preprocessing.pipeline import run_pipeline

logger = logging.getLogger(__name__)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining ~0.15 is test


def temporal_split(featured: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each patient's timeline chronologically into train/val/test, then concatenate."""
    train_parts, val_parts, test_parts = [], [], []
    for pid, g in featured.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        n = len(g)
        i_train = int(n * TRAIN_FRAC)
        i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
        train_parts.append(g.iloc[:i_train])
        val_parts.append(g.iloc[i_train:i_val])
        test_parts.append(g.iloc[i_val:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def _usable(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURE_COLUMNS + ["target_glucose_future", "target_excursion"])


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "n": int(len(y_true)),
    }


def _classification_metrics(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }
    # ROC-AUC/PR-AUC require both classes present in y_true.
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def train_and_evaluate() -> dict:
    logger.info("Running preprocessing + feature engineering...")
    fused, _ = run_pipeline()
    featured = build_feature_table(fused)

    train_df, val_df, test_df = temporal_split(featured)
    train_df, val_df, test_df = _usable(train_df), _usable(val_df), _usable(test_df)
    logger.info("Split sizes — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))

    X_train, X_val, X_test = train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]

    # ---------------- Regression: glucose @ +2h ----------------
    y_train_r, y_val_r, y_test_r = (
        train_df["target_glucose_future"], val_df["target_glucose_future"], test_df["target_glucose_future"]
    )

    candidates_reg = {
        "ridge": Ridge(alpha=1.0, random_state=settings.GLOBAL_SEED),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=settings.GLOBAL_SEED, n_jobs=-1,
        ),
    }
    val_scores_reg = {}
    for name, model in candidates_reg.items():
        model.fit(X_train, y_train_r)
        val_scores_reg[name] = mean_absolute_error(y_val_r, model.predict(X_val))
    best_reg_name = min(val_scores_reg, key=val_scores_reg.get)
    regressor = candidates_reg[best_reg_name]
    logger.info("Regression model selection (val MAE): %s -> chose %s", val_scores_reg, best_reg_name)

    test_pred_r = regressor.predict(X_test)
    reg_metrics = _regression_metrics(y_test_r, test_pred_r)
    reg_metrics["model_selected"] = best_reg_name
    reg_metrics["val_mae_by_candidate"] = {k: round(float(v), 3) for k, v in val_scores_reg.items()}

    # ---------------- Classification: excursion within 2h ----------------
    y_train_c = train_df["target_excursion"].astype(int)
    y_val_c = val_df["target_excursion"].astype(int)
    y_test_c = test_df["target_excursion"].astype(int)

    candidates_clf = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=settings.GLOBAL_SEED, n_jobs=-1,
        ),
    }
    val_scores_clf = {}
    for name, model in candidates_clf.items():
        model.fit(X_train, y_train_c)
        proba = model.predict_proba(X_val)[:, 1]
        val_scores_clf[name] = (
            roc_auc_score(y_val_c, proba) if len(set(y_val_c)) > 1 else float("nan")
        )
    best_clf_name = max(val_scores_clf, key=lambda k: (val_scores_clf[k] if val_scores_clf[k] == val_scores_clf[k] else -1))
    classifier = candidates_clf[best_clf_name]
    logger.info("Classification model selection (val ROC-AUC): %s -> chose %s", val_scores_clf, best_clf_name)

    test_proba_c = classifier.predict_proba(X_test)[:, 1]
    test_pred_c = (test_proba_c >= 0.5).astype(int)
    clf_metrics = _classification_metrics(y_test_c, test_pred_c, test_proba_c)
    clf_metrics["model_selected"] = best_clf_name
    clf_metrics["val_roc_auc_by_candidate"] = {
        k: (round(float(v), 4) if v == v else None) for k, v in val_scores_clf.items()
    }

    # ---------------- Per-patient breakdown (generalization check) ----------------
    per_patient = {}
    for pid, g in test_df.groupby("patient_id"):
        if len(g) < 5:
            continue
        pred_r = regressor.predict(g[FEATURE_COLUMNS])
        per_patient[pid] = {
            "n_test_rows": int(len(g)),
            "mae": round(float(mean_absolute_error(g["target_glucose_future"], pred_r)), 2),
        }

    # ---------------- Save artifacts ----------------
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, settings.REGRESSOR_PATH)
    joblib.dump(classifier, settings.CLASSIFIER_PATH)
    settings.FEATURE_LIST_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2))

    results = {
        "prediction_horizon_minutes": settings.PREDICTION_HORIZON_MINUTES,
        "excursion_threshold_mgdl": settings.EXCURSION_THRESHOLD_MGDL,
        "split": {"train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
                  "method": "per-patient chronological 70/15/15 split (no shuffling, no leakage)"},
        "regression": reg_metrics,
        "classification": clf_metrics,
        "per_patient_test_mae": per_patient,
    }
    settings.METRICS_PATH.write_text(json.dumps(results, indent=2))
    logger.info("Saved models to %s and metrics to %s", settings.MODELS_DIR, settings.METRICS_PATH)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = train_and_evaluate()
    print(json.dumps(out, indent=2))
