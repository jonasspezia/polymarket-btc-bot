"""
LightGBM model inference module.
Loads the serialized model plus its training metadata at startup and provides
fast prediction.
"""

import logging
import os
import pickle
from typing import Any, Optional

import lightgbm as lgb
import numpy as np

from config.settings import PATHS
from src.utils.model_metadata import (
    DEFAULT_TARGET_HORIZON_MINUTES,
    load_training_metadata,
    resolve_target_horizon_minutes,
)

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Wraps the trained LightGBM model for real-time probability predictions.
    The model is loaded once at startup and kept in memory.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path or PATHS.model_path
        self._model: Optional[lgb.Booster] = None
        self._calibrator = None  # Isotonic calibrator (Improvement 3)
        self._prediction_count = 0
        self._metadata: dict[str, Any] = {}
        self._target_horizon_minutes = DEFAULT_TARGET_HORIZON_MINUTES

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def prediction_count(self) -> int:
        return self._prediction_count

    @property
    def target_horizon_minutes(self) -> int:
        return self._target_horizon_minutes

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def load(self) -> bool:
        """
        Load the serialized LightGBM model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not os.path.exists(self._model_path):
            logger.error("Model file not found at %s", self._model_path)
            return False

        try:
            self._model = lgb.Booster(model_file=self._model_path)
            self._metadata = load_training_metadata(self._model_path)
            self._target_horizon_minutes = resolve_target_horizon_minutes(
                self._metadata
            )
            n_features = self._model.num_feature()
            n_trees = self._model.num_trees()

            # Improvement 3: Load isotonic calibrator if available
            calibrator_path = os.path.join(
                os.path.dirname(self._model_path), "calibrator.pkl"
            )
            if os.path.exists(calibrator_path):
                try:
                    with open(calibrator_path, "rb") as f:
                        self._calibrator = pickle.load(f)
                    logger.info(
                        "Isotonic calibrator loaded | path=%s", calibrator_path
                    )
                except Exception as cal_err:
                    logger.warning(
                        "Failed to load calibrator, using raw probabilities: %s",
                        cal_err,
                    )
                    self._calibrator = None
            else:
                logger.info(
                    "No calibrator found at %s — using raw model probabilities",
                    calibrator_path,
                )

            logger.info(
                "Model loaded | path=%s features=%d trees=%d target_horizon=%dm "
                "calibrated=%s",
                self._model_path,
                n_features,
                n_trees,
                self._target_horizon_minutes,
                self._calibrator is not None,
            )
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def predict(self, features: np.ndarray) -> Optional[float]:
        """
        Run inference on a single feature vector.
        
        Args:
            features: 1D numpy array of shape (n_features,)
            
        Returns:
            Probability p̂ᵢ ∈ [0, 1] that BTC goes up over the configured
            target horizon,
            or None if prediction fails.
        """
        if self._model is None:
            logger.error("Model not loaded — call load() first")
            return None

        try:
            # LightGBM expects 2D input: (n_samples, n_features)
            features_2d = features.reshape(1, -1)
            proba = self._model.predict(features_2d)
            self._prediction_count += 1

            # proba is array of shape (1,) for binary classification
            p = float(proba[0])

            # Improvement 3: Apply isotonic calibration if available
            if self._calibrator is not None:
                try:
                    p = float(self._calibrator.predict([p])[0])
                except Exception:
                    pass  # Fall back to raw probability

            if self._prediction_count % 100 == 0:
                logger.debug(
                    "Prediction #%d: p̂=%.4f", self._prediction_count, p
                )

            return p

        except Exception as e:
            logger.error("Inference error: %s", e)
            return None

    def predict_batch(self, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference on a batch of feature vectors.
        
        Args:
            features: 2D numpy array of shape (n_samples, n_features)
            
        Returns:
            Array of probabilities, or None on failure.
        """
        if self._model is None:
            return None

        try:
            return self._model.predict(features)
        except Exception as e:
            logger.error("Batch inference error: %s", e)
            return None
