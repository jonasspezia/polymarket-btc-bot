#!/usr/bin/env python3
"""
04_validate_model.py
Out-of-sample validation of the trained LightGBM model.
Generates classification report, calibration analysis, and holdout metrics.

Usage:
    python scripts/04_validate_model.py [--holdout-pct 0.2]
"""

import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PATHS
from src.features.schema import FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN
from src.utils.experiment_tracking import ExperimentTracker
from src.utils.model_metadata import (
    load_training_metadata,
    resolve_target_horizon_minutes,
    training_metadata_path_for_model,
)


def calibration_analysis(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10):
    """
    Print a calibration table showing predicted vs actual probabilities per bin.
    Well-calibrated models should show close alignment.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    print(f"\n{'='*60}")
    print("Calibration Analysis")
    print(f"{'='*60}")
    print(f"  {'Bin':>12s}  {'Pred Mean':>10s}  {'Actual Mean':>12s}  {'Count':>6s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*6}")

    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count > 0:
            pred_mean = y_proba[mask].mean()
            actual_mean = y_true[mask].mean()
            drift = actual_mean - pred_mean
            marker = "⚠" if abs(drift) > 0.05 else "✓"
            print(
                f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]"
                f"  {pred_mean:10.4f}"
                f"  {actual_mean:12.4f}"
                f"  {count:6d}"
                f"  {marker}"
            )

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Validate the trained LightGBM model")
    parser.add_argument("--holdout-pct", type=float, default=0.2, help="Holdout fraction (default: 0.2)")
    args = parser.parse_args()

    # Load data
    features_path = PATHS.features_path
    model_path = PATHS.model_path

    if not os.path.exists(features_path):
        print(f"[!] Features not found at {features_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[!] Model not found at {model_path}")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    model = lgb.Booster(model_file=model_path)
    training_metadata_path = training_metadata_path_for_model(model_path)
    training_metadata = load_training_metadata(model_path)
    target_horizon_minutes = resolve_target_horizon_minutes(training_metadata)
    experiment_id = ExperimentTracker.read_experiment_id_from_metadata(
        training_metadata_path
    )
    tracker = ExperimentTracker(experiment_id=experiment_id)

    print(f"[*] Loaded {len(df):,} samples and model from {model_path}")
    print(f"[*] Model target horizon: {target_horizon_minutes} minutes")

    # Temporal holdout split (last N% of data)
    split_idx = int(len(df) * (1 - args.holdout_pct))
    df_holdout = df.iloc[split_idx:].copy()

    # Use the model's feature columns (may be a pruned subset)
    model_feature_columns = training_metadata.get(
        "feature_columns", list(FEATURE_COLUMNS)
    )
    X_holdout = df_holdout[model_feature_columns].values
    y_holdout = df_holdout[TARGET_COLUMN].values

    print(f"[*] Holdout set: {len(df_holdout):,} samples (last {args.holdout_pct*100:.0f}%)")
    print(
        f"    Date range: "
        f"{df_holdout[TIMESTAMP_COLUMN].iloc[0]} → {df_holdout[TIMESTAMP_COLUMN].iloc[-1]}"
    )
    tracker.start_stage(
        "validation",
        label=f"validate_lightgbm_btc_{target_horizon_minutes}m",
        parameters={
            "holdout_pct": args.holdout_pct,
            "target_horizon_minutes": target_horizon_minutes,
        },
        context={
            "features_path": features_path,
            "model_path": model_path,
            "dataset_summary": ExperimentTracker.build_dataset_summary(
                df_holdout,
                timestamp_column=TIMESTAMP_COLUMN,
                target_column=TARGET_COLUMN,
            ),
        },
    )

    try:
        # Predict
        y_proba = model.predict(X_holdout)
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_holdout, y_proba)),
            "log_loss": float(log_loss(y_holdout, y_proba)),
            "accuracy": float(accuracy_score(y_holdout, y_pred)),
            "brier_score": float(brier_score_loss(y_holdout, y_proba)),
        }

        # --- Metrics ---
        print(f"\n{'='*60}")
        print("Holdout Validation Results")
        print(f"{'='*60}")
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"  Log Loss:    {metrics['log_loss']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"{'='*60}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_holdout, y_pred, target_names=["Down/Flat", "Up"]))

        # Calibration
        calibration_analysis(y_holdout, y_proba)

        # --- Edge analysis ---
        # Simulate trading performance: only trade when model probability > threshold
        print(f"\n{'='*60}")
        print("Edge Analysis (Simulated Trading Signal)")
        print(f"{'='*60}")

        edge_analysis = []
        for threshold in [0.50, 0.52, 0.55, 0.58, 0.60]:
            strong_signal = y_proba >= threshold
            if strong_signal.sum() > 0:
                precision = y_holdout[strong_signal].mean()
                n_trades = strong_signal.sum()
                pct_time = n_trades / len(y_holdout) * 100
                print(
                    f"  p̂ ≥ {threshold:.2f}: "
                    f"Precision={precision:.4f}  "
                    f"Trades={n_trades:,}  "
                    f"({pct_time:.1f}% of time)"
                )
                edge_analysis.append({
                    "threshold": threshold,
                    "precision": float(precision),
                    "trades": int(n_trades),
                    "pct_time": float(pct_time),
                })

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "experiment_id": tracker.experiment_id,
            "model_path": str(model_path),
            "target_horizon_minutes": target_horizon_minutes,
            "holdout_samples": len(df_holdout),
            "holdout_pct": args.holdout_pct,
            "holdout_date_range": {
                "start": str(df_holdout[TIMESTAMP_COLUMN].iloc[0]),
                "end": str(df_holdout[TIMESTAMP_COLUMN].iloc[-1]),
            },
            "metrics": metrics,
            "edge_analysis": edge_analysis,
        }

        report_path = os.path.join(os.path.dirname(model_path), "validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        artifacts = [
            tracker.copy_artifact(report_path, "reports/validation_report.json"),
        ]
        tracker.complete_stage(
            "validation",
            summary={
                "metrics": metrics,
                "holdout_samples": len(df_holdout),
                "holdout_pct": args.holdout_pct,
                "target_horizon_minutes": target_horizon_minutes,
            },
            artifacts=artifacts,
        )

        print(f"[*] Experiment ID: {tracker.experiment_id}")
        print(f"[*] Experiment directory: {tracker.experiment_dir}")
        print(f"\n[✓] Validation complete. Report saved to {report_path}")
    except Exception as e:
        tracker.fail_stage("validation", str(e))
        raise


if __name__ == "__main__":
    main()
