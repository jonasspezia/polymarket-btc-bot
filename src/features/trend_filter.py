"""
EMA trend confirmation filter.

Improvement 2: Blocks entries that disagree with the short-term trend
determined by EMA crossover. This cuts counter-trend losses by
preventing the bot from fighting momentum.
"""

import logging
from typing import Optional

import numpy as np

from src.utils.state import RollingState

logger = logging.getLogger(__name__)


class TrendFilter:
    """
    Fast EMA / Slow EMA crossover gate.

    Only allows entries when the trade direction agrees with the
    short-term trend. This prevents buying YES during downtrends
    and buying NO during uptrends.
    """

    def __init__(
        self,
        fast_period: int = 8,
        slow_period: int = 21,
        min_prices: int = 25,
    ):
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._min_prices = max(min_prices, slow_period + 1)

    def confirms_direction(self, side: str, state: RollingState) -> bool:
        """
        Return True when the EMA crossover agrees with the trade side.

        - BUY_YES: requires fast EMA > slow EMA (uptrend)
        - BUY_NO: requires fast EMA < slow EMA (downtrend)

        Returns True (permissive) when insufficient data is available.
        """
        prices = state.get_prices()
        if len(prices) < self._min_prices:
            return True  # Not enough data — be permissive

        fast_ema = self._ema(prices, self._fast_period)
        slow_ema = self._ema(prices, self._slow_period)

        if fast_ema is None or slow_ema is None:
            return True

        is_uptrend = fast_ema > slow_ema

        if side == "BUY_YES":
            if not is_uptrend:
                logger.debug(
                    "Trend filter blocked BUY_YES | fast_ema=%.2f slow_ema=%.2f",
                    fast_ema,
                    slow_ema,
                )
                return False
            return True

        if side == "BUY_NO":
            if is_uptrend:
                logger.debug(
                    "Trend filter blocked BUY_NO | fast_ema=%.2f slow_ema=%.2f",
                    fast_ema,
                    slow_ema,
                )
                return False
            return True

        return True

    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> Optional[float]:
        """Compute the EMA of the last `period` prices."""
        if len(prices) < period:
            return None

        alpha = 2.0 / (period + 1)
        ema = float(prices[0])
        for price in prices[1:]:
            ema = alpha * float(price) + (1 - alpha) * ema
        return ema
