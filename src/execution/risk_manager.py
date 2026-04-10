"""
Risk management module.
Implements volatility kill-switch, mass cancellation, position limits, and P&L floor.
"""

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from config.settings import RISK, TRADING
from src.exchange.polymarket_client import PolymarketClient
from src.utils.state import RollingState

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Monitors trading risk in real-time and triggers protective actions:
    
    1. Volatility Kill-Switch: Cancels all orders if BTC vol spikes beyond threshold
    2. Position Limits: Prevents accumulating too many open positions
    3. P&L Floor: Halts trading if cumulative losses breach the floor
    4. Cooldown: Pauses trading after a kill-switch event
    """

    def __init__(
        self,
        state: RollingState,
        client: PolymarketClient,
        read_only_mode: bool = False,
        sigma_threshold: Optional[float] = None,
        min_absolute_volatility: Optional[float] = None,
        min_relative_volatility_multiplier: Optional[float] = None,
        cooldown_seconds: Optional[int] = None,
        pnl_floor: Optional[float] = None,
        max_positions: Optional[int] = None,
        private_check_cache_ttl_seconds: Optional[float] = None,
        min_available_collateral: Optional[float] = None,
        max_available_collateral_drawdown: Optional[float] = None,
    ):
        self._state = state
        self._client = client
        self._sigma_threshold = (
            sigma_threshold
            if sigma_threshold is not None
            else RISK.volatility_sigma_threshold
        )
        self._min_absolute_volatility = (
            min_absolute_volatility
            if min_absolute_volatility is not None
            else RISK.volatility_min_absolute_threshold
        )
        self._min_relative_volatility_multiplier = (
            min_relative_volatility_multiplier
            if min_relative_volatility_multiplier is not None
            else RISK.volatility_min_relative_multiplier
        )
        self._cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else RISK.kill_switch_cooldown_seconds
        )
        self._pnl_floor = pnl_floor if pnl_floor is not None else RISK.pnl_floor
        self._max_positions = (
            max_positions
            if max_positions is not None
            else TRADING.max_open_positions
        )
        self._private_check_cache_ttl_seconds = (
            private_check_cache_ttl_seconds
            if private_check_cache_ttl_seconds is not None
            else RISK.private_check_cache_ttl_seconds
        )
        self._min_available_collateral = (
            min_available_collateral
            if min_available_collateral is not None
            else getattr(RISK, "min_available_collateral", 0.0)
        )
        self._max_available_collateral_drawdown = (
            max_available_collateral_drawdown
            if max_available_collateral_drawdown is not None
            else getattr(RISK, "max_available_collateral_drawdown", 0.0)
        )
        self._read_only_mode = read_only_mode

        # Internal state
        self._is_killed = False
        self._kill_time: Optional[float] = None
        self._kill_count = 0
        self._cumulative_pnl = 0.0
        self._trading_halted = False

        # Volatility baseline tracking
        self._vol_history: deque[float] = deque(
            maxlen=max(30, RISK.vol_lookback_window)
        )
        self._last_vol_sample_timestamp_ms: Optional[int] = None

        # Cache noisy private account reads for a short TTL.
        self._last_position_limit_check_monotonic = 0.0
        self._cached_position_limit_result: Optional[bool] = None
        self._cached_open_order_count: Optional[int] = None
        self._last_balance_check_monotonic = 0.0
        self._cached_balance_check_result: Optional[bool] = None
        self._cached_collateral_status = None
        self._starting_available_collateral: Optional[float] = None
        self._lowest_available_collateral: Optional[float] = None

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        if self._trading_halted:
            return False

        if self._is_killed:
            # Check if cooldown has elapsed
            if self._kill_time and (time.time() - self._kill_time) >= self._cooldown_seconds:
                logger.info("Kill-switch cooldown elapsed — resuming trading")
                self._is_killed = False
                self._kill_time = None
                return True
            return False

        return True

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    @property
    def kill_count(self) -> int:
        return self._kill_count

    @property
    def cumulative_pnl(self) -> float:
        return self._cumulative_pnl

    def check_volatility(self) -> bool:
        """
        Monitor volatility and trigger kill-switch if threshold is breached.
        
        Returns:
            True if kill-switch was triggered, False otherwise.
        """
        latest_timestamp_ms = self._state.latest_timestamp_ms
        if latest_timestamp_ms <= 0:
            return False

        if latest_timestamp_ms == self._last_vol_sample_timestamp_ms:
            return False

        current_vol = self._state.get_volatility(30)  # 30-second realized vol
        
        if current_vol <= 0:
            return False

        baseline_samples = np.array(self._vol_history, dtype=np.float64)

        # Track volatility history
        self._vol_history.append(current_vol)
        self._last_vol_sample_timestamp_ms = latest_timestamp_ms

        if len(baseline_samples) < 30:
            return False  # Not enough history to establish baseline

        # Compute baseline statistics
        vol_mean = float(np.mean(baseline_samples))
        vol_std = float(np.std(baseline_samples))

        if vol_mean <= 0 or vol_std < 1e-12:
            return False

        # Z-score of current volatility
        z_score = (current_vol - vol_mean) / vol_std
        relative_multiplier = current_vol / vol_mean

        if z_score >= self._sigma_threshold:
            if current_vol < self._min_absolute_volatility:
                logger.debug(
                    "Volatility spike ignored below absolute floor | current_vol=%.6f "
                    "baseline_vol=%.6f z_score=%.2f abs_floor=%.6f",
                    current_vol,
                    vol_mean,
                    z_score,
                    self._min_absolute_volatility,
                )
                return False

            if relative_multiplier < self._min_relative_volatility_multiplier:
                logger.debug(
                    "Volatility spike ignored below relative floor | current_vol=%.6f "
                    "baseline_vol=%.6f rel_multiplier=%.2fx min_multiplier=%.2fx "
                    "z_score=%.2f",
                    current_vol,
                    vol_mean,
                    relative_multiplier,
                    self._min_relative_volatility_multiplier,
                    z_score,
                )
                return False

            logger.critical(
                "[KILL_SWITCH] VOLATILITY KILL-SWITCH TRIGGERED | "
                "current_vol=%.6f baseline_vol=%.6f rel_multiplier=%.2fx "
                "z_score=%.2f threshold=%.1fσ | trading paused for %d seconds",
                current_vol,
                vol_mean,
                relative_multiplier,
                z_score,
                self._sigma_threshold,
                self._cooldown_seconds,
            )
            self._trigger_kill_switch()
            return True

        return False

    def check_position_limit(self) -> bool:
        """
        Check if we've reached the maximum number of open positions.
        
        Returns:
            True if under limit (trading ok), False if at limit.
        """
        if self._has_fresh_position_limit_cache():
            return bool(self._cached_position_limit_result)

        try:
            open_orders = self._client.get_open_orders()
            n_open = len(open_orders) if open_orders else 0

            result = n_open < self._max_positions
            self._cached_open_order_count = n_open
            self._cached_position_limit_result = result
            self._last_position_limit_check_monotonic = time.monotonic()

            if not result:
                logger.warning(
                    "Position limit reached | open=%d max=%d",
                    n_open, self._max_positions,
                )
                return False
            return True
        except Exception as e:
            logger.error("Failed to check positions: %s", e)
            self._cached_open_order_count = None
            self._cached_position_limit_result = False
            self._last_position_limit_check_monotonic = time.monotonic()
            return False  # Fail closed on position-check failure

    def check_pnl_floor(self) -> bool:
        """
        Check if cumulative P&L has breached the floor.
        
        Returns:
            True if above floor (trading ok), False if breached.
        """
        if self._cumulative_pnl <= self._pnl_floor:
            logger.critical(
                "P&L FLOOR BREACHED | pnl=%.4f floor=%.4f — HALTING TRADING",
                self._cumulative_pnl, self._pnl_floor,
            )
            self._trading_halted = True
            self._trigger_kill_switch()
            return False
        return True

    def update_pnl(self, realized_pnl: float):
        """
        Update cumulative P&L with a realized trade result.
        
        Args:
            realized_pnl: Positive for profit, negative for loss.
        """
        self._cumulative_pnl += realized_pnl
        logger.info(
            "P&L updated | trade=%.4f cumulative=%.4f floor=%.4f",
            realized_pnl, self._cumulative_pnl, self._pnl_floor,
        )

    def check_balance(self) -> bool:
        """
        Perform a coarse collateral preflight before live order placement.

        Exact order-cost enforcement still lives in the router, because the
        router knows the candidate limit price and resolved size.

        Returns:
            True if balance is sufficient, False otherwise.
        """
        if self._has_fresh_balance_cache():
            return bool(self._cached_balance_check_result)

        status = self._client.get_collateral_balance_allowance()
        self._cached_collateral_status = status
        self._last_balance_check_monotonic = time.monotonic()

        if status is None:
            logger.error("Unable to verify collateral balance/allowance")
            self._cached_balance_check_result = False
            return False

        if status.available_to_trade <= 0:
            logger.warning(
                "No spendable collateral available | balance=%.6f allowance=%.6f",
                status.balance,
                status.allowance,
            )
            self._cached_balance_check_result = False
            return False

        available_to_trade = float(status.available_to_trade)
        self._update_collateral_baseline(available_to_trade)

        if (
            self._min_available_collateral > 0
            and available_to_trade + 1e-9 < self._min_available_collateral
        ):
            logger.critical(
                "Available collateral below live safety floor | available=%.4f min_required=%.4f",
                available_to_trade,
                self._min_available_collateral,
            )
            self._cached_balance_check_result = False
            self._trading_halted = True
            self._trigger_kill_switch()
            return False

        if (
            self._starting_available_collateral is not None
            and self._max_available_collateral_drawdown > 0
        ):
            drawdown = self._starting_available_collateral - available_to_trade
            if drawdown > self._max_available_collateral_drawdown + 1e-9:
                logger.critical(
                    "Available collateral drawdown breached | baseline=%.4f current=%.4f drawdown=%.4f max_drawdown=%.4f",
                    self._starting_available_collateral,
                    available_to_trade,
                    drawdown,
                    self._max_available_collateral_drawdown,
                )
                self._cached_balance_check_result = False
                self._trading_halted = True
                self._trigger_kill_switch()
                return False

        self._cached_balance_check_result = True
        return True

    def _update_collateral_baseline(self, available_to_trade: float) -> None:
        """Track the session collateral baseline for drawdown guardrails."""
        if self._starting_available_collateral is None:
            self._starting_available_collateral = available_to_trade
            self._lowest_available_collateral = available_to_trade
            logger.info(
                "Collateral baseline initialized | available=%.4f min_floor=%.4f max_drawdown=%.4f",
                available_to_trade,
                self._min_available_collateral,
                self._max_available_collateral_drawdown,
            )
            return

        if self._lowest_available_collateral is None:
            self._lowest_available_collateral = available_to_trade
            return

        self._lowest_available_collateral = min(
            self._lowest_available_collateral,
            available_to_trade,
        )

    def run_all_checks(
        self,
        include_balance_check: bool = True,
        include_position_limit: bool = True,
    ) -> bool:
        """
        Run all risk checks. Returns True if trading is allowed.
        """
        if not self.is_trading_allowed:
            return False

        if not self.check_pnl_floor():
            return False

        if self.check_volatility():
            return False

        if include_position_limit and not self.check_position_limit():
            return False

        if include_balance_check and not self.check_balance():
            return False

        return True

    def _trigger_kill_switch(self):
        """Execute the kill-switch: cancel all orders and enter cooldown."""
        self._is_killed = True
        self._kill_time = time.time()
        self._kill_count += 1
        self.invalidate_private_check_cache()

        if self._read_only_mode:
            logger.warning(
                "Kill-switch #%d: Read-only mode active — skipping cancel_all_orders | "
                "cooldown=%ds",
                self._kill_count,
                self._cooldown_seconds,
            )
            return

        # Mass cancel all resting orders
        success = self._client.cancel_all_orders()
        
        if success:
            logger.warning(
                "Kill-switch #%d: All orders cancelled | cooldown=%ds",
                self._kill_count, self._cooldown_seconds,
            )
        else:
            logger.error(
                "Kill-switch #%d: Cancel attempt FAILED — manual intervention may be needed",
                self._kill_count,
            )

    def invalidate_private_check_cache(self):
        """Force the next private account checks to refetch from Polymarket."""
        self._last_position_limit_check_monotonic = 0.0
        self._cached_position_limit_result = None
        self._cached_open_order_count = None
        self._last_balance_check_monotonic = 0.0
        self._cached_balance_check_result = None
        self._cached_collateral_status = None

    def _has_fresh_position_limit_cache(self) -> bool:
        if self._private_check_cache_ttl_seconds <= 0:
            return False
        if self._cached_position_limit_result is None:
            return False
        age_seconds = time.monotonic() - self._last_position_limit_check_monotonic
        return age_seconds < self._private_check_cache_ttl_seconds

    def _has_fresh_balance_cache(self) -> bool:
        if self._private_check_cache_ttl_seconds <= 0:
            return False
        if self._cached_balance_check_result is None:
            return False
        age_seconds = time.monotonic() - self._last_balance_check_monotonic
        return age_seconds < self._private_check_cache_ttl_seconds

    def get_status(self) -> dict:
        """Get current risk manager status as a dict."""
        return {
            "trading_allowed": self.is_trading_allowed,
            "halted": self._trading_halted,
            "killed": self._is_killed,
            "kill_count": self._kill_count,
            "cumulative_pnl": self._cumulative_pnl,
            "pnl_floor": self._pnl_floor,
            "cooldown_remaining": max(
                0,
                self._cooldown_seconds - (time.time() - self._kill_time)
                if self._kill_time else 0,
            ),
            "min_available_collateral": self._min_available_collateral,
            "max_available_collateral_drawdown": (
                self._max_available_collateral_drawdown
            ),
            "starting_available_collateral": self._starting_available_collateral,
            "lowest_available_collateral": self._lowest_available_collateral,
            "vol_history_size": len(self._vol_history),
            "cached_open_order_count": self._cached_open_order_count,
        }
