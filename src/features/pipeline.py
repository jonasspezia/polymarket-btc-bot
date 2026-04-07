"""
Real-time feature computation pipeline.
Reads from the rolling trade buffer, synthesizes minute bars, and computes the
same feature vector used during offline training.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.features.minute_features import (
    BAR_INPUT_COLUMNS,
    aggregate_trades_to_1m_bars,
    compute_feature_frame,
)
from src.features.schema import FEATURE_COLUMNS
from src.utils.state import RollingState

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Computes the real-time feature vector from the rolling trade state.
    """

    def __init__(self, state: RollingState, model_feature_columns: list[str] = None):
        self._state = state
        self._model_feature_columns = model_feature_columns or list(FEATURE_COLUMNS)
        self._feature_count = len(self._model_feature_columns)
        self._historical_bars = pd.DataFrame(columns=BAR_INPUT_COLUMNS)

    @property
    def feature_names(self) -> list[str]:
        return list(self._model_feature_columns)

    @property
    def feature_count(self) -> int:
        return self._feature_count

    @property
    def required_complete_bars(self) -> int:
        """
        Number of closed 1-minute bars required before all model features are finite.
        """
        return 102

    @property
    def minute_bar_count(self) -> int:
        """Current number of closed minute bars available to the pipeline."""
        return len(self._build_minute_bar_frame())

    @property
    def is_ready(self) -> bool:
        """Whether the pipeline has enough closed minute bars for inference."""
        return self.minute_bar_count >= self.required_complete_bars

    def seed_historical_bars(self, bars: pd.DataFrame):
        """Seed the pipeline with recent closed minute bars from REST history."""
        if bars.empty:
            self._historical_bars = pd.DataFrame(columns=BAR_INPUT_COLUMNS)
            return

        seeded = bars[list(BAR_INPUT_COLUMNS)].copy()
        seeded = seeded.sort_values("open_time").drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        seeded.reset_index(drop=True, inplace=True)
        self._historical_bars = seeded

    def compute(self) -> Optional[np.ndarray]:
        """
        Compute the full feature vector from the current rolling state.

        Returns:
            A 1D numpy array matching the model feature schema, or None when we
            do not yet have enough completed minute bars to reproduce training.
        """
        if not self.is_ready:
            return None

        try:
            bars = self._build_minute_bar_frame()
            if bars.empty:
                return None

            feature_frame = compute_feature_frame(bars)
            if feature_frame.empty:
                return None

            latest = feature_frame.iloc[-1]
            features = latest[self._model_feature_columns].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(features)):
                return None

            return features

        except Exception as e:
            logger.error("Feature computation error: %s", e)
            return None

    def _build_minute_bar_frame(self) -> pd.DataFrame:
        """Combine REST-seeded history with closed live minute bars."""
        live_bars = aggregate_trades_to_1m_bars(self._state.get_trades())

        if self._historical_bars.empty and live_bars.empty:
            return pd.DataFrame(columns=BAR_INPUT_COLUMNS)
        if self._historical_bars.empty:
            return live_bars
        if live_bars.empty:
            return self._historical_bars.copy()

        combined = pd.concat(
            [self._historical_bars, live_bars],
            ignore_index=True,
        )
        combined = combined.sort_values("open_time").drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        combined.reset_index(drop=True, inplace=True)
        return combined
