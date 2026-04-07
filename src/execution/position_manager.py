"""
Manage open positions and dynamic exit rules.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from config.settings import TRADING
from src.exchange.gamma_api import GammaAPIClient, MarketInfo
from src.exchange.polymarket_client import OrderResult, PolymarketClient
from src.execution.market_rules import derive_market_resolution_rule, is_position_favorable
from src.execution.order_router import TradingSignal
from src.utils.state import RollingState

logger = logging.getLogger(__name__)


@dataclass
class ManagedPosition:
    """Tracked entry used for simulated or live dynamic exits."""
    condition_id: str
    market_slug: str
    market_question: str
    token_id: str
    side: str
    entry_price: float
    initial_size: float
    remaining_size: float
    end_date: str
    neg_risk: bool
    entry_timestamp: float
    take_profit_taken: bool = False


@dataclass
class ExitExecution:
    """Outcome of an attempted exit rule execution."""
    position_key: str
    reason: str
    exit_price: float
    exit_size: float
    realized_pnl: float
    remaining_size: float
    result: OrderResult


class PositionManager:
    """Track open positions and apply defensive exit rules."""

    def __init__(
        self,
        client: PolymarketClient,
        read_only_mode: bool = False,
        stop_loss_factor: Optional[float] = None,
        take_profit_multiple: Optional[float] = None,
        take_profit_fraction: Optional[float] = None,
        time_decay_exit_seconds: Optional[int] = None,
        time_decay_distance_pct: Optional[float] = None,
        dynamic_exits_enabled: Optional[bool] = None,
    ):
        self._client = client
        self._read_only_mode = read_only_mode
        self._dynamic_exits_enabled = (
            dynamic_exits_enabled
            if dynamic_exits_enabled is not None
            else getattr(TRADING, "enable_dynamic_exits", True)
        )
        self._stop_loss_factor = (
            stop_loss_factor
            if stop_loss_factor is not None
            else getattr(TRADING, "stop_loss_factor", 0.5)
        )
        self._take_profit_multiple = (
            take_profit_multiple
            if take_profit_multiple is not None
            else getattr(TRADING, "take_profit_multiple", 3.0)
        )
        self._take_profit_fraction = (
            take_profit_fraction
            if take_profit_fraction is not None
            else getattr(TRADING, "take_profit_fraction", 0.5)
        )
        self._time_decay_exit_seconds = (
            time_decay_exit_seconds
            if time_decay_exit_seconds is not None
            else getattr(TRADING, "time_decay_exit_seconds", 1800)
        )
        self._time_decay_distance_pct = (
            time_decay_distance_pct
            if time_decay_distance_pct is not None
            else getattr(TRADING, "time_decay_distance_pct", 0.005)
        )
        self._positions: dict[str, ManagedPosition] = {}

    def record_entry(self, signal: TradingSignal, market: MarketInfo):
        """Track a new filled or simulated entry."""
        position_key = self._position_key(market.condition_id, signal.side)
        existing = self._positions.get(position_key)
        if existing is None:
            self._positions[position_key] = ManagedPosition(
                condition_id=market.condition_id,
                market_slug=market.slug,
                market_question=market.question,
                token_id=signal.token_id,
                side=signal.side,
                entry_price=signal.price,
                initial_size=signal.size,
                remaining_size=signal.size,
                end_date=market.end_date,
                neg_risk=market.neg_risk,
                entry_timestamp=signal.timestamp,
            )
            return

        total_size = existing.remaining_size + signal.size
        if total_size <= 0:
            return
        existing.entry_price = (
            (existing.entry_price * existing.remaining_size)
            + (signal.price * signal.size)
        ) / total_size
        existing.initial_size += signal.size
        existing.remaining_size = total_size
        existing.end_date = market.end_date
        existing.neg_risk = market.neg_risk

    def evaluate_positions(self, state: RollingState) -> list[ExitExecution]:
        """Run exit rules against all tracked positions."""
        if not self._dynamic_exits_enabled:
            return []

        executions: list[ExitExecution] = []
        for position_key, position in list(self._positions.items()):
            decision = self._choose_exit(position, state)
            if decision is None:
                continue
            execution = self._execute_exit(position_key, position, decision)
            if execution is None:
                continue
            executions.append(execution)
        return executions

    def get_status(self) -> dict:
        """Return a compact position summary for monitoring."""
        return {
            "managed_positions": len(self._positions),
            "managed_position_keys": sorted(self._positions.keys()),
        }

    def _choose_exit(self, position: ManagedPosition, state: RollingState) -> Optional[dict]:
        current_bid, current_ask = self._client.get_best_bid_ask(position.token_id)
        mark_price = current_bid if current_bid is not None and current_bid > 0 else None
        if mark_price is None:
            return None

        # Bug A fix: `stop_loss_factor` is the max fraction of entry price
        # you're willing to lose. E.g. factor=0.5 on a 0.50 entry means
        # stop at 0.50 * (1 - 0.5) = 0.25, i.e. you accept losing $0.25.
        stop_loss_price = position.entry_price * (1.0 - self._stop_loss_factor)
        if mark_price <= stop_loss_price:
            return {
                "reason": "stop_loss",
                "exit_size": position.remaining_size,
                "mark_price": mark_price,
                "limit_price": self._exit_limit_price(current_bid, current_ask),
            }

        # Bug B fix: on binary markets, prices cap at 1.0 so
        # `entry * 3.0` is unreachable for any entry > 0.33.
        # Reinterpret as: TP triggers when unrealized gain reaches
        # `take_profit_multiple` times the entry edge (1.0 - entry).
        # E.g. entry=0.50, edge=0.50, multiple=0.3 → TP at 0.50+0.15=0.65
        tp_edge = (1.0 - position.entry_price) * self._take_profit_multiple
        take_profit_price = min(position.entry_price + tp_edge, 0.99)
        if (
            not position.take_profit_taken
            and self._take_profit_multiple > 0
            and mark_price >= take_profit_price
        ):
            exit_size = position.remaining_size * min(max(self._take_profit_fraction, 0.0), 1.0)
            if exit_size > 0:
                return {
                    "reason": "take_profit_scale",
                    "exit_size": min(exit_size, position.remaining_size),
                    "mark_price": mark_price,
                    "limit_price": self._exit_limit_price(current_bid, current_ask),
                }

        decay_decision = self._time_decay_exit(position, state, mark_price, current_bid, current_ask)
        if decay_decision is not None:
            return decay_decision

        return None

    def _time_decay_exit(
        self,
        position: ManagedPosition,
        state: RollingState,
        mark_price: float,
        current_bid: Optional[float],
        current_ask: Optional[float],
    ) -> Optional[dict]:
        end_ts = GammaAPIClient._parse_iso_timestamp(position.end_date)
        if end_ts is None:
            return None

        now_ts = state.latest_timestamp_ms / 1000.0 if state.latest_timestamp_ms > 0 else time.time()
        time_remaining = end_ts - now_ts
        if time_remaining > self._time_decay_exit_seconds:
            return None

        spot_price = state.last_price
        if spot_price <= 0:
            return None

        rule = derive_market_resolution_rule(
            MarketInfo(
                condition_id=position.condition_id,
                question=position.market_question,
                slug=position.market_slug,
                yes_token_id="",
                no_token_id="",
                end_date=position.end_date,
                neg_risk=position.neg_risk,
            )
        )
        if rule.reference_price is None or rule.reference_price <= 0:
            return None

        favorable = is_position_favorable(rule, position.side, spot_price)
        if favorable is not False:
            return None

        distance_pct = abs(spot_price - rule.reference_price) / rule.reference_price
        if distance_pct < self._time_decay_distance_pct:
            return None

        return {
            "reason": "time_decay_exit",
            "exit_size": position.remaining_size,
            "mark_price": mark_price,
            "limit_price": self._exit_limit_price(current_bid, current_ask),
        }

    def _execute_exit(
        self,
        position_key: str,
        position: ManagedPosition,
        decision: dict,
    ) -> Optional[ExitExecution]:
        exit_size = float(decision["exit_size"])
        if exit_size <= 0:
            return None

        mark_price = float(decision["mark_price"])
        realized_pnl = (mark_price - position.entry_price) * exit_size

        if self._read_only_mode:
            result = OrderResult(
                success=True,
                order_id=f"dry-run-exit:{position.condition_id}:{decision['reason']}",
                raw_response={
                    "dry_run": True,
                    "exit_reason": decision["reason"],
                    "price": mark_price,
                    "size": exit_size,
                    "realized_pnl": realized_pnl,
                },
            )
        else:
            result = self._client.place_post_only_gtd(
                token_id=position.token_id,
                price=float(decision["limit_price"]),
                size=exit_size,
                side="SELL",
                neg_risk=position.neg_risk,
            )
            if result.raw_response is None:
                result.raw_response = {}
            result.raw_response["exit_reason"] = decision["reason"]
            result.raw_response["estimated_exit_price"] = mark_price
            result.raw_response["estimated_realized_pnl"] = realized_pnl

        if not result.success:
            logger.warning(
                "Exit order failed | market=%s side=%s reason=%s",
                position.market_slug,
                position.side,
                decision["reason"],
            )
            return ExitExecution(
                position_key=position_key,
                reason=decision["reason"],
                exit_price=mark_price,
                exit_size=0.0,
                realized_pnl=0.0,
                remaining_size=position.remaining_size,
                result=result,
            )

        position.remaining_size = max(position.remaining_size - exit_size, 0.0)
        if decision["reason"] == "take_profit_scale":
            position.take_profit_taken = True

        if position.remaining_size <= 1e-9:
            self._positions.pop(position_key, None)

        logger.info(
            "Position exit executed | market=%s side=%s reason=%s price=%.2f size=%.4f "
            "remaining=%.4f pnl=%.4f",
            position.market_slug,
            position.side,
            decision["reason"],
            mark_price,
            exit_size,
            position.remaining_size,
            realized_pnl,
        )
        return ExitExecution(
            position_key=position_key,
            reason=decision["reason"],
            exit_price=mark_price,
            exit_size=exit_size,
            realized_pnl=realized_pnl,
            remaining_size=position.remaining_size,
            result=result,
        )

    @staticmethod
    def _position_key(condition_id: str, side: str) -> str:
        return f"{condition_id}:{side}"

    @staticmethod
    def _exit_limit_price(current_bid: Optional[float], current_ask: Optional[float]) -> float:
        tick = float(TRADING.tick_size)
        if current_ask is not None and current_ask > 0:
            return min(round(current_ask, 2), 0.99)
        if current_bid is not None and current_bid > 0:
            return min(round(current_bid + tick, 2), 0.99)
        return 0.99
