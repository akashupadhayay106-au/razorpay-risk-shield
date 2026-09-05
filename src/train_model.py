"""
End-to-end model training for the Razorpay Risk Shield fraud detector.

Pipeline: load data -> EDA -> stratified 70/15/15 split -> feature engineering
(one-hot + scaling) -> cost-sensitive XGBoost tuned with GridSearchCV ->
evaluation (precision/recall/F1/FPR/ROC-AUC) -> cost simulation vs. a rule
baseline -> persist artifacts under ``models/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FeatureExtractor  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Cost model (Rs): a false negative (missed fraud / chargeback) costs 5x the
# cost of a false positive (lost legitimate transaction).
FP_COST = 500
FN_COST = 2500

RANDOM_STATE = 42


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("\n=== EDA Summary ===")
    print("Shape:", df.shape)
    print("\nClass distribution:")
    print(df["is_fraud"].value_counts(normalize=True))
    print("\nMissing values (top):\n", df.isnull().sum())
    print("\nNumeric stats:\n", df.describe().T.loc[:, ["mean", "std", "min", "max"]])

    y = df["is_fraud"].values
    X_df = df.drop(columns=["is_fraud"])

    # ------------------------------------------------------------------
    # 70 / 15 / 15 stratified split
    # ------------------------------------------------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_df, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=(0.15 / 0.85),
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )
    print(f"\nSplits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Preprocessing: one-hot encoding + scaling (fit on train only)
    # ------------------------------------------------------------------
    print("\nPreprocessing...")
    extractor = FeatureExtractor()
    extractor.fit(X_train)
    print(f"n_features: {len(extractor.get_feature_names())}")

    X_train_scaled = extractor.transform(X_train)
    X_val_scaled = extractor.transform(X_val)
    X_test_scaled = extractor.transform(X_test)

    # ------------------------------------------------------------------
    # Cost-sensitive learning
    # ------------------------------------------------------------------
    count_neg = int(np.sum(y_train == 0))
    count_pos = int(np.sum(y_train == 1))
    # scale_pos_weight amplifies the minority (fraud) class by the FP/FN cost ratio.
    scale_pos_weight = (count_neg / count_pos) * (FN_COST / FP_COST)
    print(f"\nscale_pos_weight = (neg/pos) * (FN/FP) = {scale_pos_weight:.2f}")

    # ------------------------------------------------------------------
    # XGBoost + GridSearchCV
    #   - use_label_encoder is REMOVED (deprecated in xgboost >= 2.0)
    #   - eval_metric='logloss' is set explicitly
    # ------------------------------------------------------------------
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
    }

    print("\nRunning GridSearchCV (cv=3, n_jobs=-1)...")
    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1",
        verbose=1,
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    print("\nBest parameters:", grid.best_params_)
    print(f"Best CV F1: {grid.best_score_:.4f}")

    # ------------------------------------------------------------------
    # Evaluate on held-out test set
    # ------------------------------------------------------------------
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n=== Metrics (Test Set) ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"CM: TN={tn} FP={fp} FN={fn} TP={tp}")

    # ------------------------------------------------------------------
    # Cost simulation vs. rule-based baseline
    #   Baseline: decline ALL transactions > Rs50,000 OR foreign IP.
    # ------------------------------------------------------------------
    baseline_decline = (X_test["amount"] > 50_000) | (X_test["is_foreign_ip"] == 1)
    baseline_fp = int(np.sum(baseline_decline & (y_test == 0)))
    baseline_fn = int(np.sum(~baseline_decline & (y_test == 1)))
    baseline_cost = baseline_fp * FP_COST + baseline_fn * FN_COST

    ml_cost = fp * FP_COST + fn * FN_COST
    cost_saved = baseline_cost - ml_cost

    print("\n=== Cost Simulation (Rs) ===")
    print(f"Baseline rule cost (FP={baseline_fp}, FN={baseline_fn}): Rs{baseline_cost:,.0f}")
    print(f"ML model cost     (FP={fp}, FN={fn}): Rs{ml_cost:,.0f}")
    print(f"Total Cost Saved (Rs): Rs{cost_saved:,.0f}")

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    print("\nSaving artifacts to", MODELS_DIR)
    joblib.dump(best_model, MODELS_DIR / "risk_model.pkl")
    joblib.dump(extractor.scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(extractor.get_feature_names(), MODELS_DIR / "feature_names.pkl")
    joblib.dump(extractor, MODELS_DIR / "extractor.pkl")

    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    baseline_cost = int(baseline_cost)
    ml_cost = int(ml_cost)
    cost_saved = baseline_cost - ml_cost

    metrics = {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "cm": [[tn, fp], [fn, tp]],
        "threshold": 0.5,
        "fp_cost": FP_COST,
        "fn_cost": FN_COST,
        "total_cost_baseline": baseline_cost,
        "total_cost_ml": ml_cost,
        "total_cost_saved": cost_saved,
        "best_params": grid.best_params_,
    }
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nDone! All artifacts + metrics saved.")


if __name__ == "__main__":
    main()