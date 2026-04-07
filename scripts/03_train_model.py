#!/usr/bin/env python3
"""
03_train_model.py
Thorough LightGBM training pipeline with:
  1. Purged time-series CV (gap between train/val to prevent leakage)
  2. Optuna hyperparameter optimization (100 trials)
  3. Noise feature pruning
  4. Multi-seed ensemble for robustness
  5. Isotonic calibration
  6. Comprehensive reporting

Usage:
    python scripts/03_train_model.py [--optuna-trials 100] [--n-splits 5]
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Optional

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

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PATHS
from src.features.schema import FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN
from src.utils.experiment_tracking import ExperimentTracker



# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Purged Time-Series Cross-Validation
# ---------------------------------------------------------------------------

class PurgedTimeSeriesCV:
    """
    Time-series CV with an embargo gap between train and validation sets.
    This prevents information leakage through overlapping features/targets.
    
    The purge gap ensures no training sample's target overlaps with the
    validation set's feature lookback window.
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 10):
        self.n_splits = n_splits
        self.purge_gap = purge_gap  # Number of rows to skip between train/val

    def split(self, X):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n)
            if val_start >= n or val_end <= val_start:
                continue
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 3,
) -> list[str]:
    """
    Remove noise features using a quick LightGBM importance filter.
    Features with zero average gain across folds are dropped.
    """
    print("[*] Running feature importance pre-screening...")
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "feature_fraction": 1.0,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        "min_data_in_leaf": 100,
    }

    cv = PurgedTimeSeriesCV(n_splits=n_splits, purge_gap=10)
    importances = np.zeros(X.shape[1])
    n_folds = 0

    for train_idx, val_idx in cv.split(X):
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feature_names)
        dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feature_names, reference=dtrain)

        model = lgb.train(
            params, dtrain, num_boost_round=100,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )
        importances += model.feature_importance(importance_type="gain")
        n_folds += 1

    importances /= max(n_folds, 1)
    
    # Keep features with non-trivial importance (> 1% of max)
    threshold = importances.max() * 0.01
    selected = [name for name, imp in zip(feature_names, importances) if imp >= threshold]
    dropped = [name for name, imp in zip(feature_names, importances) if imp < threshold]

    if dropped:
        print(f"    Dropping {len(dropped)} noise features: {dropped}")
    print(f"    Keeping {len(selected)} features: {selected}")
    return selected


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Optimization
# ---------------------------------------------------------------------------

def run_optuna_search(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    n_trials: int = 100,
    purge_gap: int = 10,
) -> dict:
    """
    Run Optuna hyperparameter search over purged CV.
    Optimizes for AUC on out-of-sample folds.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = PurgedTimeSeriesCV(n_splits=n_splits, purge_gap=purge_gap)
    # Pre-compute splits once
    splits = list(cv.split(X))

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            # --- Tuned hyperparameters ---
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        }

        fold_aucs = []
        for train_idx, val_idx in splits:
            dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feature_names)
            dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feature_names, reference=dtrain)

            model = lgb.train(
                params, dtrain, num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            preds = model.predict(X[val_idx], num_iteration=model.best_iteration)
            auc = roc_auc_score(y[val_idx], preds)
            fold_aucs.append(auc)

        return np.mean(fold_aucs)

    study = optuna.create_study(direction="maximize", study_name="lgbm_btc")
    
    # Start with a known-good baseline
    study.enqueue_trial({
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "min_gain_to_split": 0.0,
        "lambda_l1": 1e-5,
        "lambda_l2": 1e-5,
        "max_depth": 6,
        "path_smooth": 0.0,
    })

    print(f"\n[*] Running Optuna search ({n_trials} trials, {n_splits}-fold purged CV)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n[✓] Best trial #{best.number}: AUC={best.value:.6f}")
    print("    Best parameters:")
    for k, v in best.params.items():
        print(f"      {k}: {v}")

    return best.params


# ---------------------------------------------------------------------------
# Multi-Seed Training
# ---------------------------------------------------------------------------

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    best_params: dict,
    n_splits: int = 5,
    purge_gap: int = 10,
    seeds: list[int] = None,
) -> tuple[lgb.Booster, list[dict], IsotonicRegression, dict]:
    """
    Train the final model with best hyperparameters.
    Uses multi-seed ensemble for robustness.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 2024]

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": -1,
        **best_params,
    }

    cv = PurgedTimeSeriesCV(n_splits=n_splits, purge_gap=purge_gap)
    splits = list(cv.split(X))

    print(f"\n{'='*60}")
    print(f"Final Training — {len(seeds)} seeds × {len(splits)} folds")
    print(f"{'='*60}")

    # --- Cross-validation with best params ---
    all_fold_metrics = []
    best_iterations = []

    for fold_i, (train_idx, val_idx) in enumerate(splits, 1):
        fold_aucs = []
        fold_best_iters = []

        for seed in seeds:
            params_s = {**params, "seed": seed}
            dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feature_names)
            dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feature_names, reference=dtrain)

            model = lgb.train(
                params_s, dtrain, num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
            preds = model.predict(X[val_idx], num_iteration=model.best_iteration)
            fold_aucs.append(roc_auc_score(y[val_idx], preds))
            fold_best_iters.append(model.best_iteration)

        avg_auc = np.mean(fold_aucs)
        avg_iter = int(np.mean(fold_best_iters))
        best_iterations.append(avg_iter)

        # Compute full metrics with primary seed
        params_primary = {**params, "seed": seeds[0]}
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feature_names)
        dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feature_names, reference=dtrain)
        model = lgb.train(
            params_primary, dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X[val_idx], num_iteration=model.best_iteration)
        y_pred = (preds >= 0.5).astype(int)

        metrics = {
            "fold": fold_i,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "auc": avg_auc,
            "auc_std": float(np.std(fold_aucs)),
            "log_loss": log_loss(y[val_idx], preds),
            "accuracy": accuracy_score(y[val_idx], y_pred),
            "brier_score": brier_score_loss(y[val_idx], preds),
            "best_iteration": avg_iter,
        }
        all_fold_metrics.append(metrics)
        print(
            f"  Fold {fold_i}/{len(splits)}: "
            f"AUC={metrics['auc']:.4f}±{metrics['auc_std']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  "
            f"Brier={metrics['brier_score']:.4f}  "
            f"(iter={metrics['best_iteration']})"
        )

    # --- Summary ---
    metrics_df = pd.DataFrame(all_fold_metrics)
    print(f"\n{'='*60}")
    print("Cross-Validation Summary (purged CV, multi-seed)")
    print(f"{'='*60}")
    for metric in ["auc", "log_loss", "accuracy", "brier_score"]:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        print(f"  {metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
    print(f"{'='*60}")

    # --- Train final ensemble on full data ---
    avg_best_iter = int(np.mean(best_iterations))
    # Use 10% more iterations on the full dataset since no early stopping
    final_iters = max(int(avg_best_iter * 1.1), avg_best_iter + 5)
    print(f"\n[*] Retraining on full dataset with {final_iters} iterations ({len(seeds)} seeds)...")

    full_dataset = lgb.Dataset(X, label=y, feature_name=feature_names)

    # Train multiple models with different seeds for ensemble averaging
    final_models = []
    for seed in seeds:
        params_s = {**params, "seed": seed}
        m = lgb.train(params_s, full_dataset, num_boost_round=final_iters)
        final_models.append(m)

    # Primary model is seed[0] — saved as the main model file
    primary_model = final_models[0]

    # --- Isotonic calibration ---
    last_train_idx, last_val_idx = splits[-1]
    
    # Ensemble predictions on calibration set
    cal_preds_all = np.column_stack([
        m.predict(X[last_val_idx]) for m in final_models
    ])
    cal_preds = cal_preds_all.mean(axis=1)
    y_cal = y[last_val_idx]

    calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    calibrator.fit(cal_preds, y_cal)
    
    cal_calibrated = calibrator.predict(cal_preds)
    brier_before = brier_score_loss(y_cal, cal_preds)
    brier_after = brier_score_loss(y_cal, cal_calibrated)
    print(f"[✓] Isotonic calibrator fitted")
    print(f"    Brier score: {brier_before:.4f} → {brier_after:.4f} "
          f"({'improved' if brier_after < brier_before else 'no change'})")

    ensemble_info = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "final_iterations": final_iters,
        "avg_cv_best_iteration": avg_best_iter,
    }

    return primary_model, all_fold_metrics, calibrator, ensemble_info, final_models


# ---------------------------------------------------------------------------
# Save Model + Artifacts
# ---------------------------------------------------------------------------

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
    ensemble_info: dict[str, Any],
    best_params: dict[str, Any],
    ensemble_models: list[lgb.Booster] = None,
    selected_features: list[str] = None,
) -> list[dict[str, Any]]:
    """Serialize model, ensemble, calibrator and metadata."""
    os.makedirs(PATHS.models_dir, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat()

    # Save primary model
    model_path = PATHS.model_path
    model.save_model(model_path)
    print(f"[✓] Model saved to {model_path}")

    # Save ensemble models
    if ensemble_models:
        ensemble_dir = os.path.join(PATHS.models_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        for i, m in enumerate(ensemble_models):
            m.save_model(os.path.join(ensemble_dir, f"model_seed_{i}.txt"))
        print(f"[✓] Ensemble models saved to {ensemble_dir}/ ({len(ensemble_models)} models)")

    # Save isotonic calibrator
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
        "feature_columns": list(selected_features or FEATURE_COLUMNS),
        "n_features": len(selected_features or FEATURE_COLUMNS),
        "fold_metrics": metrics,
        "cv_summary": cv_summary,
        "training_parameters": training_parameters,
        "best_hyperparameters": best_params,
        "ensemble_info": ensemble_info,
        "dataset_summary": dataset_summary,
        "model_file": PATHS.model_filename,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✓] Training metadata saved to {meta_path}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_names = selected_features or list(FEATURE_COLUMNS)
    feat_imp = sorted(
        zip(feat_names, importance), key=lambda x: x[1], reverse=True
    )
    feature_importance_payload = [
        {"feature": name, "gain": float(imp)} for name, imp in feat_imp
    ]
    print("\n[*] Feature Importance (gain):")
    max_imp = max(importance) if len(importance) > 0 else 1
    for name, imp in feat_imp:
        bar = "█" * int(imp / max_imp * 30)
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
                "best_hyperparameters": best_params,
                "ensemble_info": ensemble_info,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Thoroughly train LightGBM for BTC prediction")
    parser.add_argument("--n-splits", type=int, default=5, help="Purged CV folds")
    parser.add_argument("--purge-gap", type=int, default=10, help="Rows to skip between train/val")
    parser.add_argument("--optuna-trials", type=int, default=100, help="Optuna optimization trials")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Optuna, use defaults")
    parser.add_argument("--skip-feature-selection", action="store_true", help="Skip feature pruning")
    parser.add_argument("--n-seeds", type=int, default=5, help="Seeds for ensemble")
    parser.add_argument(
        "--target-horizon-minutes", type=int, default=5,
        help="Target horizon used when dataset was engineered",
    )
    args = parser.parse_args()

    df = load_features()
    tracker = ExperimentTracker()

    feature_names = list(FEATURE_COLUMNS)
    X = df[feature_names].values
    y = df[TARGET_COLUMN].values

    print(f"\n{'='*60}")
    print(f"Thorough Training Pipeline")
    print(f"{'='*60}")
    print(f"  Samples:         {len(X):,}")
    print(f"  Features:        {len(feature_names)}")
    print(f"  Target balance:  {np.mean(y):.4f}")
    print(f"  CV Folds:        {args.n_splits}")
    print(f"  Purge gap:       {args.purge_gap} rows")
    print(f"  Optuna trials:   {args.optuna_trials}")
    print(f"  Ensemble seeds:  {args.n_seeds}")
    print(f"{'='*60}")

    # --- Step 1: Feature selection ---
    selected_features = feature_names
    if not args.skip_feature_selection:
        selected_features = select_features(X, y, feature_names, n_splits=3)
        # Re-slice X to selected features only
        feat_indices = [feature_names.index(f) for f in selected_features]
        X = X[:, feat_indices]
    else:
        print("[*] Feature selection skipped")

    # --- Step 2: Hyperparameter optimization ---
    if args.skip_optuna:
        best_params = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 50,
            "min_gain_to_split": 0.0,
            "lambda_l1": 1e-5,
            "lambda_l2": 1e-5,
            "max_depth": 6,
            "path_smooth": 0.0,
        }
        print("[*] Optuna search skipped — using default parameters")
    else:
        best_params = run_optuna_search(
            X, y, selected_features,
            n_splits=args.n_splits,
            n_trials=args.optuna_trials,
            purge_gap=args.purge_gap,
        )

    # --- Step 3: Final training with multi-seed ensemble ---
    seeds = list(range(42, 42 + args.n_seeds))
    model, metrics, calibrator, ensemble_info, ensemble_models = train_final_model(
        X, y, selected_features,
        best_params=best_params,
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
        seeds=seeds,
    )

    # --- Step 4: Save everything ---
    training_parameters = {
        "n_splits": args.n_splits,
        "purge_gap": args.purge_gap,
        "optuna_trials": args.optuna_trials,
        "n_seeds": args.n_seeds,
        "target_horizon_minutes": args.target_horizon_minutes,
    }
    dataset_summary = ExperimentTracker.build_dataset_summary(
        df, timestamp_column=TIMESTAMP_COLUMN, target_column=TARGET_COLUMN,
    )
    dataset_summary["target_horizon_minutes"] = args.target_horizon_minutes

    tracker.start_stage(
        "training",
        label=f"train_lightgbm_btc_{args.target_horizon_minutes}m_thorough",
        parameters=training_parameters,
        context={
            "features_path": PATHS.features_path,
            "model_output_path": PATHS.model_path,
            "dataset_summary": dataset_summary,
        },
    )

    try:
        artifacts = save_model(
            model, metrics, calibrator,
            experiment_id=tracker.experiment_id,
            target_horizon_minutes=args.target_horizon_minutes,
            training_parameters=training_parameters,
            dataset_summary=dataset_summary,
            tracker=tracker,
            ensemble_info=ensemble_info,
            best_params=best_params,
            ensemble_models=ensemble_models,
            selected_features=selected_features,
        )
        tracker.complete_stage(
            "training",
            summary={
                "cv_summary": ExperimentTracker.summarize_fold_metrics(metrics),
                "rows": dataset_summary["rows"],
                "model_path": PATHS.model_path,
                "target_horizon_minutes": args.target_horizon_minutes,
                "best_hyperparameters": best_params,
                "ensemble_seeds": seeds,
            },
            artifacts=artifacts,
        )
    except Exception as e:
        tracker.fail_stage("training", str(e))
        raise

    print(f"\n[*] Experiment ID: {tracker.experiment_id}")
    print(f"[*] Experiment directory: {tracker.experiment_dir}")
    print("\n[✓] Thorough training complete.")


if __name__ == "__main__":
    main()
