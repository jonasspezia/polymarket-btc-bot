#!/usr/bin/env python3
"""
03_train_model.py
Trains a LightGBM binary classifier for BTC price direction prediction.
Uses TimeSeriesSplit for k-fold cross-validation to respect temporal ordering.

Usage:
    python scripts/03_train_model.py [--n-splits 5] [--n-estimators 500]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PATHS
from src.features.schema import FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN
from src.utils.experiment_tracking import ExperimentTracker


def load_features() -> pd.DataFrame:
    """Load the engineered feature dataset."""
    path = PATHS.features_path
    if not os.path.exists(path):
        print(f"[!] Features file not found at {path}")
        print("    Run scripts/02_engineer_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"[*] Loaded {len(df):,} samples from {path}")
    return df


def train_with_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.Booster, list[dict[str, Any]]]:
    """
    Train LightGBM with TimeSeriesSplit cross-validation.
    Returns the best model trained on the final fold.
    """
    X = df[list(FEATURE_COLUMNS)].values
    y = df[TARGET_COLUMN].values

    print(f"\n{'='*60}")
    print(f"LightGBM Training Configuration")
    print(f"{'='*60}")
    print(f"  Samples:              {len(X):,}")
    print(f"  Features:             {len(FEATURE_COLUMNS)}")
    print(f"  Target balance:       {np.mean(y):.4f} (frac positive)")
    print(f"  CV Splits:            {n_splits}")
    print(f"  Estimators:           {n_estimators}")
    print(f"  Learning rate:        {learning_rate}")
    print(f"  Num leaves:           {num_leaves}")
    print(f"  Early stopping:       {early_stopping_rounds}")
    print(f"{'='*60}\n")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=list(FEATURE_COLUMNS)
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            feature_name=list(FEATURE_COLUMNS),
            reference=train_data,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Predict probabilities
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        metrics = {
            "fold": fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "auc": roc_auc_score(y_val, y_pred_proba),
            "log_loss": log_loss(y_val, y_pred_proba),
            "accuracy": accuracy_score(y_val, y_pred),
            "brier_score": brier_score_loss(y_val, y_pred_proba),
            "best_iteration": model.best_iteration,
        }
        fold_metrics.append(metrics)

        print(
            f"  Fold {fold}/{n_splits}: "
            f"AUC={metrics['auc']:.4f}  "
            f"LogLoss={metrics['log_loss']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  "
            f"Brier={metrics['brier_score']:.4f}  "
            f"(iter={metrics['best_iteration']})"
        )
    # --- Summary ---
    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    for metric in ["auc", "log_loss", "accuracy", "brier_score"]:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        print(f"  {metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
    print(f"{'='*60}\n")

    # --- Retrain on full dataset with best iteration count ---
    avg_best_iter = int(metrics_df["best_iteration"].mean())
    print(f"[*] Retraining on full dataset with {avg_best_iter} iterations...")

    full_train = lgb.Dataset(X, label=y, feature_name=list(FEATURE_COLUMNS))
    final_model = lgb.train(
        params,
        full_train,
        num_boost_round=avg_best_iter,
    )

    # --- Improvement 3: Isotonic calibration on last fold ---
    # Use the last fold's out-of-sample predictions to fit a calibrator
    last_train_idx, last_val_idx = list(tscv.split(X))[-1]
    X_cal_val = X[last_val_idx]
    y_cal_val = y[last_val_idx]
    y_cal_pred = final_model.predict(X_cal_val)

    calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    calibrator.fit(y_cal_pred, y_cal_val)
    print("[✓] Isotonic calibrator fitted on last fold out-of-sample predictions")

    # Verify calibration improvement
    y_cal_calibrated = calibrator.predict(y_cal_pred)
    brier_before = brier_score_loss(y_cal_val, y_cal_pred)
    brier_after = brier_score_loss(y_cal_val, y_cal_calibrated)
    print(
        f"    Brier score: {brier_before:.4f} → {brier_after:.4f} "
        f"({'improved' if brier_after < brier_before else 'no change'})"
    )

    return final_model, fold_metrics, calibrator


def save_model(
    model: lgb.Booster,
    metrics: list[dict[str, Any]],
    calibrator: IsotonicRegression,
    *,
    experiment_id: str,
    target_horizon_minutes: int,
    training_parameters: dict[str, Any],
    dataset_summary: dict[str, Any],
    tracker: ExperimentTracker,
) -> list[dict[str, Any]]:
    """Serialize model and save training metadata."""
    os.makedirs(PATHS.models_dir, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat()

    # Save model
    model_path = PATHS.model_path
    model.save_model(model_path)
    print(f"[✓] Model saved to {model_path}")

    # Save isotonic calibrator
    import pickle
    calibrator_path = os.path.join(PATHS.models_dir, "calibrator.pkl")
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"[✓] Isotonic calibrator saved to {calibrator_path}")

    # Save metadata
    meta_path = os.path.join(PATHS.models_dir, "training_metadata.json")
    cv_summary = ExperimentTracker.summarize_fold_metrics(metrics)
    metadata = {
        "trained_at": trained_at,
        "experiment_id": experiment_id,
        "target_horizon_minutes": target_horizon_minutes,
        "feature_columns": list(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        "fold_metrics": metrics,
        "cv_summary": cv_summary,
        "training_parameters": training_parameters,
        "dataset_summary": dataset_summary,
        "model_file": PATHS.model_filename,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✓] Training metadata saved to {meta_path}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(
        zip(FEATURE_COLUMNS, importance), key=lambda x: x[1], reverse=True
    )
    feature_importance_payload = [
        {"feature": name, "gain": float(imp)}
        for name, imp in feat_imp
    ]
    print("\n[*] Feature Importance (gain):")
    for name, imp in feat_imp:
        bar = "█" * int(imp / max(importance) * 30)
        print(f"    {name:25s} {imp:10.1f}  {bar}")

    artifacts = [
        tracker.copy_artifact(model_path, f"model/{os.path.basename(model_path)}"),
        tracker.copy_artifact(meta_path, "model/training_metadata.json"),
        tracker.write_json_artifact(
            "reports/training_summary.json",
            {
                "trained_at": trained_at,
                "experiment_id": experiment_id,
                "target_horizon_minutes": target_horizon_minutes,
                "dataset_summary": dataset_summary,
                "training_parameters": training_parameters,
                "cv_summary": cv_summary,
                "fold_metrics": metrics,
            },
        ),
        tracker.write_json_artifact(
            "reports/feature_importance.json",
            {
                "experiment_id": experiment_id,
                "importance_type": "gain",
                "features": feature_importance_payload,
            },
        ),
    ]
    return artifacts


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model for BTC prediction")
    parser.add_argument("--n-splits", type=int, default=5, help="TimeSeriesSplit folds")
    parser.add_argument("--n-estimators", type=int, default=500, help="Max boosting rounds")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--num-leaves", type=int, default=31, help="Num leaves per tree")
    parser.add_argument(
        "--target-horizon-minutes",
        type=int,
        default=5,
        help="Target horizon used when the feature dataset was engineered",
    )
    args = parser.parse_args()

    df = load_features()
    tracker = ExperimentTracker()
    training_parameters = {
        "n_splits": args.n_splits,
        "n_estimators": args.n_estimators,
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "target_horizon_minutes": args.target_horizon_minutes,
    }
    dataset_summary = ExperimentTracker.build_dataset_summary(
        df,
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
    )
    dataset_summary["target_horizon_minutes"] = args.target_horizon_minutes
    tracker.start_stage(
        "training",
        label=f"train_lightgbm_btc_{args.target_horizon_minutes}m",
        parameters=training_parameters,
        context={
            "features_path": PATHS.features_path,
            "model_output_path": PATHS.model_path,
            "dataset_summary": dataset_summary,
        },
    )

    try:
        model, metrics, calibrator = train_with_cv(
            df,
            n_splits=args.n_splits,
            n_estimators=args.n_estimators,
            learning_rate=args.lr,
            num_leaves=args.num_leaves,
        )

        artifacts = save_model(
            model,
            metrics,
            calibrator,
            experiment_id=tracker.experiment_id,
            target_horizon_minutes=args.target_horizon_minutes,
            training_parameters=training_parameters,
            dataset_summary=dataset_summary,
            tracker=tracker,
        )
        tracker.complete_stage(
            "training",
            summary={
                "cv_summary": ExperimentTracker.summarize_fold_metrics(metrics),
                "rows": dataset_summary["rows"],
                "model_path": PATHS.model_path,
                "target_horizon_minutes": args.target_horizon_minutes,
            },
            artifacts=artifacts,
        )
    except Exception as e:
        tracker.fail_stage("training", str(e))
        raise

    print(f"[*] Experiment ID: {tracker.experiment_id}")
    print(f"[*] Experiment directory: {tracker.experiment_dir}")
    print("\n[✓] Training complete.")


if __name__ == "__main__":
    main()
