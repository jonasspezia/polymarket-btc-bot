from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.exchange.gamma_api import MarketInfo
from src.execution.order_router import TradingSignal
from src.execution.position_manager import PositionManager
from src.utils.state import RollingState


def _push_trade(state: RollingState, price: float, timestamp_ms: int, trade_id: int):
    state.push_trade_sync(
        {
            "price": price,
            "quantity": 1.0,
            "timestamp": timestamp_ms,
            "is_buyer_maker": False,
            "trade_id": trade_id,
        }
    )


def _market(end_date: str, question: str = "Bitcoin above 105 on April 6, 11AM ET?"):
    return MarketInfo(
        condition_id="market-1",
        question=question,
        slug="bitcoin-above-105-on-april-6-2026-11am-et",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date=end_date,
    )


def _signal(side: str = "BUY_YES", price: float = 0.11, size: float = 10.0):
    return TradingSignal(
        side=side,
        token_id="yes-token" if side == "BUY_YES" else "no-token",
        price=price,
        size=size,
        edge=0.05,
        model_prob=0.55,
        market_price=price + 0.01,
        timestamp=1.0,
    )


def test_position_manager_triggers_stop_loss_in_read_only_mode():
    client = MagicMock()
    client.get_best_bid_ask.return_value = (0.05, 0.06)
    manager = PositionManager(client, read_only_mode=True, stop_loss_factor=0.5)
    end_date = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manager.record_entry(_signal(price=0.11, size=10.0), _market(end_date=end_date))
    state = RollingState(maxlen=5)
    _push_trade(state, 100.0, 0, 1)
    _push_trade(state, 99.8, 1_000, 2)

    executions = manager.evaluate_positions(state)

    assert len(executions) == 1
    assert executions[0].reason == "stop_loss"
    assert executions[0].exit_size == pytest.approx(10.0)
    assert executions[0].realized_pnl == pytest.approx((0.05 - 0.11) * 10.0)
    assert executions[0].remaining_size == 0.0


def test_position_manager_scales_out_on_take_profit():
    client = MagicMock()
    # New TP formula: TP = entry + (1-entry)*multiple
    # entry=0.11, multiple=0.3 → TP = 0.11 + 0.89*0.3 = 0.377
    # Set bid=0.38 to trigger TP
    client.get_best_bid_ask.return_value = (0.38, 0.39)
    manager = PositionManager(
        client,
        read_only_mode=True,
        take_profit_multiple=0.3,
        take_profit_fraction=0.5,
    )
    end_date = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manager.record_entry(_signal(price=0.11, size=10.0), _market(end_date=end_date))
    state = RollingState(maxlen=5)
    _push_trade(state, 100.0, 0, 1)
    _push_trade(state, 100.1, 1_000, 2)

    executions = manager.evaluate_positions(state)

    assert len(executions) == 1
    assert executions[0].reason == "take_profit_scale"
    assert executions[0].exit_size == pytest.approx(5.0)
    assert executions[0].remaining_size == pytest.approx(5.0)


def test_position_manager_triggers_time_decay_exit_for_bad_threshold_setup():
    client = MagicMock()
    client.get_best_bid_ask.return_value = (0.08, 0.09)
    manager = PositionManager(
        client,
        read_only_mode=True,
        time_decay_exit_seconds=1800,
        time_decay_distance_pct=0.005,
    )
    latest_ts = int(datetime(2026, 4, 6, 14, 55, tzinfo=timezone.utc).timestamp() * 1000)
    end_date = datetime(2026, 4, 6, 15, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    manager.record_entry(_signal(price=0.11, size=10.0), _market(end_date=end_date))
    state = RollingState(maxlen=5)
    _push_trade(state, 99.0, latest_ts - 1_000, 1)
    _push_trade(state, 99.0, latest_ts, 2)

    executions = manager.evaluate_positions(state)

    assert len(executions) == 1
    assert executions[0].reason == "time_decay_exit"
    assert executions[0].exit_size == pytest.approx(10.0)
