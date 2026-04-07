"""
Market-aware probability estimation.

The trained model predicts short-horizon BTC direction. This module adapts that
directional signal into a market-specific YES probability so threshold markets
like "Bitcoin above 69,800 at 3PM ET" aren't treated like simple up/down bets.
"""

import math
import time
from statistics import NormalDist
from typing import Optional

import numpy as np

from config.settings import TRADING
from src.exchange.gamma_api import GammaAPIClient, MarketInfo
from src.execution.market_rules import derive_market_resolution_rule
from src.utils.state import RollingState


class MarketProbabilityEstimator:
    """Translate a directional model output into a market YES probability."""

    def __init__(
        self,
        strategy_style: Optional[str] = None,
        vol_window_seconds: Optional[int] = None,
        min_sigma: Optional[float] = None,
    ):
        self._strategy_style = (
            strategy_style
            if strategy_style is not None
            else getattr(TRADING, "strategy_style", "momentum")
        ).strip().lower()
        self._vol_window_seconds = (
            vol_window_seconds
            if vol_window_seconds is not None
            else getattr(TRADING, "probability_vol_window_seconds", 300)
        )
        self._min_sigma = (
            min_sigma
            if min_sigma is not None
            else getattr(TRADING, "probability_min_sigma", 0.0015)
        )

    def estimate_yes_probability(
        self,
        directional_up_probability: float,
        market: MarketInfo,
        state: RollingState,
    ) -> float:
        """
        Estimate the probability that YES resolves true for the given market.
        """
        raw_prob = self._clamp_probability(directional_up_probability)
        rule = derive_market_resolution_rule(market)
        if rule.resolution_type == "move":
            return raw_prob

        spot_price = state.last_price
        if spot_price <= 0 or rule.reference_price is None or rule.reference_price <= 0:
            return raw_prob

        sigma_horizon = self._estimate_horizon_sigma(state, market)
        if sigma_horizon <= 0:
            return raw_prob

        directional_z = NormalDist().inv_cdf(raw_prob)
        directional_z = self._apply_strategy_bias(directional_z, state)
        mu_horizon = sigma_horizon * directional_z

        target_log_return = math.log(rule.reference_price / spot_price)
        distribution = NormalDist(mu=mu_horizon, sigma=sigma_horizon)

        if rule.resolution_type == "above":
            yes_prob = 1.0 - distribution.cdf(target_log_return)
        elif rule.resolution_type == "below":
            yes_prob = distribution.cdf(target_log_return)
        else:
            yes_prob = raw_prob

        return self._clamp_probability(yes_prob)

    def _estimate_horizon_sigma(
        self,
        state: RollingState,
        market: MarketInfo,
    ) -> float:
        """Estimate the log-return sigma between now and market expiry."""
        recent_trades = state.get_window_by_time(self._vol_window_seconds)
        if len(recent_trades) < 3:
            return self._min_sigma

        prices = np.array([trade.price for trade in recent_trades], dtype=np.float64)
        log_returns = np.log(prices[1:] / prices[:-1])
        trade_sigma = float(np.std(log_returns))
        if not math.isfinite(trade_sigma) or trade_sigma <= 0:
            return self._min_sigma

        trades_per_second = max((len(recent_trades) - 1) / self._vol_window_seconds, 1 / 60)
        now_ts = (
            state.latest_timestamp_ms / 1000.0
            if state.latest_timestamp_ms > 0
            else time.time()
        )
        end_ts = GammaAPIClient._parse_iso_timestamp(market.end_date)
        if end_ts is None:
            return self._min_sigma

        time_remaining_seconds = max(end_ts - now_ts, 60.0)
        expected_trade_steps = max(trades_per_second * time_remaining_seconds, 1.0)
        sigma_horizon = trade_sigma * math.sqrt(expected_trade_steps)
        return max(sigma_horizon, self._min_sigma)

    def _apply_strategy_bias(self, directional_z: float, state: RollingState) -> float:
        """
        Nudge the model's direction score based on the configured style.

        - momentum: gently reinforce recent directional drift
        - mean_reversion: gently fade recent directional drift

        Bug 4 fix: reduced multiplier from 0.25 to 0.10 to avoid
        double-counting the model's own momentum features (momentum_1m
        through momentum_10m already capture recent drift).
        """
        latest_ts = state.latest_timestamp_ms
        if latest_ts <= 0:
            return directional_z

        lookback_price = state.get_price_at_or_before(latest_ts - 60_000)
        current_price = state.last_price
        if lookback_price is None or lookback_price <= 0 or current_price <= 0:
            return directional_z

        recent_return = math.log(current_price / lookback_price)
        recent_scale = max(self._estimate_recent_sigma(state), self._min_sigma)
        recent_z = max(min(recent_return / recent_scale, 2.0), -2.0)

        bias_strength = 0.10  # Reduced from 0.25 to avoid double-counting

        if self._strategy_style == "mean_reversion":
            return directional_z - (bias_strength * recent_z)
        return directional_z + (bias_strength * recent_z)

    def _estimate_recent_sigma(self, state: RollingState) -> float:
        recent_trades = state.get_window_by_time(60)
        if len(recent_trades) < 3:
            return self._min_sigma

        prices = np.array([trade.price for trade in recent_trades], dtype=np.float64)
        log_returns = np.log(prices[1:] / prices[:-1])
        sigma = float(np.std(log_returns))
        return sigma if math.isfinite(sigma) and sigma > 0 else self._min_sigma

    @staticmethod
    def _clamp_probability(value: float) -> float:
        if not math.isfinite(value):
            return 0.5
        return min(max(value, 1e-4), 1 - 1e-4)
