"""Tests for the risk manager module."""

import time
from unittest.mock import MagicMock
from types import SimpleNamespace

import numpy as np
import pytest

from src.execution.risk_manager import RiskManager
from src.utils.state import RollingState


class _SequencedVolatilityState:
    def __init__(self, vol_samples):
        self._vol_samples = list(vol_samples)
        self._index = -1
        self._current_vol = 0.0
        self._latest_timestamp_ms = 0

    @property
    def latest_timestamp_ms(self) -> int:
        return self._latest_timestamp_ms

    def advance(self):
        self._index += 1
        self._current_vol = self._vol_samples[self._index]
        self._latest_timestamp_ms += 1000

    def get_volatility(self, window_seconds: int) -> float:
        return self._current_vol


@pytest.fixture
def state():
    """Create a RollingState with some trade data."""
    s = RollingState(maxlen=500)
    base_time = int(time.time() * 1000)
    # Add 200 trades with normal volatility
    for i in range(200):
        price = 95000 + np.random.randn() * 10  # Low vol: ±$10
        s.push_trade_sync({
            "price": price,
            "quantity": 0.001,
            "timestamp": base_time + i * 100,  # 100ms apart
            "is_buyer_maker": bool(np.random.random() > 0.5),
            "trade_id": i,
        })
    return s


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.cancel_all_orders.return_value = True
    client.get_open_orders.return_value = []
    client.get_collateral_balance_allowance.return_value = SimpleNamespace(
        balance=10.0,
        allowance=10.0,
        available_to_trade=10.0,
    )
    return client


@pytest.fixture
def risk_manager(state, mock_client):
    return RiskManager(
        state=state,
        client=mock_client,
        sigma_threshold=3.0,
        cooldown_seconds=5,
        pnl_floor=-0.20,
        max_positions=3,
        private_check_cache_ttl_seconds=60.0,
        min_available_collateral=10.0,
        max_available_collateral_drawdown=1.0,
    )


class TestRiskManager:
    def test_initial_state(self, risk_manager):
        assert risk_manager.is_trading_allowed
        assert not risk_manager.is_halted
        assert risk_manager.kill_count == 0
        assert risk_manager.cumulative_pnl == 0.0

    def test_pnl_floor_breach(self, risk_manager, mock_client):
        """Trading should halt when P&L drops below the floor."""
        risk_manager.update_pnl(-0.25)
        assert not risk_manager.check_pnl_floor()
        assert risk_manager.is_halted
        assert not risk_manager.is_trading_allowed
        mock_client.cancel_all_orders.assert_called()

    def test_pnl_tracking(self, risk_manager):
        risk_manager.update_pnl(0.05)
        assert risk_manager.cumulative_pnl == pytest.approx(0.05)
        risk_manager.update_pnl(-0.02)
        assert risk_manager.cumulative_pnl == pytest.approx(0.03)

    def test_position_limit(self, risk_manager, mock_client):
        """Should block trading when at max positions."""
        mock_client.get_open_orders.return_value = [
            {"id": "1"}, {"id": "2"}, {"id": "3"}
        ]
        assert not risk_manager.check_position_limit()

    def test_position_under_limit(self, risk_manager, mock_client):
        mock_client.get_open_orders.return_value = [{"id": "1"}]
        assert risk_manager.check_position_limit()

    def test_position_check_failure_blocks_trading(self, risk_manager, mock_client):
        """Live safety should fail closed if open-order state can't be checked."""
        mock_client.get_open_orders.side_effect = RuntimeError("API unavailable")
        assert not risk_manager.check_position_limit()

    def test_position_limit_uses_cached_private_result(
        self, risk_manager, mock_client
    ):
        mock_client.get_open_orders.return_value = [{"id": "1"}]

        assert risk_manager.check_position_limit()
        assert risk_manager.check_position_limit()
        mock_client.get_open_orders.assert_called_once()

    def test_check_balance_blocks_when_no_spendable_collateral(
        self, risk_manager, mock_client
    ):
        """Balance preflight should fail closed when allowance leaves nothing spendable."""
        mock_client.get_collateral_balance_allowance.return_value = SimpleNamespace(
            balance=1.0,
            allowance=0.0,
            available_to_trade=0.0,
        )
        assert not risk_manager.check_balance()

    def test_check_balance_halts_when_available_collateral_drops_below_floor(
        self, risk_manager, mock_client
    ):
        mock_client.get_collateral_balance_allowance.side_effect = [
            SimpleNamespace(balance=10.5, allowance=10.5, available_to_trade=10.5),
            SimpleNamespace(balance=9.9, allowance=9.9, available_to_trade=9.9),
        ]

        assert risk_manager.check_balance()
        risk_manager.invalidate_private_check_cache()
        assert not risk_manager.check_balance()
        assert risk_manager.is_halted

    def test_check_balance_halts_on_session_drawdown(
        self, risk_manager, mock_client
    ):
        mock_client.get_collateral_balance_allowance.side_effect = [
            SimpleNamespace(balance=12.0, allowance=12.0, available_to_trade=12.0),
            SimpleNamespace(balance=10.8, allowance=10.8, available_to_trade=10.8),
        ]

        assert risk_manager.check_balance()
        risk_manager.invalidate_private_check_cache()
        assert not risk_manager.check_balance()
        assert risk_manager.is_halted

    def test_check_balance_blocks_when_collateral_status_is_unavailable(
        self, risk_manager, mock_client
    ):
        """Balance preflight should fail closed when private account state can't be read."""
        mock_client.get_collateral_balance_allowance.return_value = None
        assert not risk_manager.check_balance()

    def test_balance_check_uses_cached_private_result(self, risk_manager, mock_client):
        assert risk_manager.check_balance()
        assert risk_manager.check_balance()
        mock_client.get_collateral_balance_allowance.assert_called_once()

    def test_kill_switch_cooldown(self, risk_manager):
        """Kill-switch should block trading for the cooldown duration."""
        risk_manager._trigger_kill_switch()
        assert not risk_manager.is_trading_allowed
        assert risk_manager.kill_count == 1

    def test_kill_switch_recovery(self, risk_manager):
        """Trading should resume after cooldown expires."""
        risk_manager._trigger_kill_switch()
        assert not risk_manager.is_trading_allowed

        # Simulate time passing beyond cooldown
        risk_manager._kill_time = time.time() - 10  # 10s ago (cooldown is 5s)
        assert risk_manager.is_trading_allowed

    def test_status_report(self, risk_manager):
        status = risk_manager.get_status()
        assert "trading_allowed" in status
        assert "halted" in status
        assert "cumulative_pnl" in status
        assert "kill_count" in status

    def test_run_all_checks_normal(self, risk_manager):
        """All checks should pass under normal conditions."""
        assert risk_manager.run_all_checks()

    def test_run_all_checks_can_skip_position_limit(self, risk_manager, mock_client):
        """Read-only validation should not require open-order limit checks."""
        mock_client.get_open_orders.side_effect = RuntimeError("API unavailable")

        assert risk_manager.run_all_checks(include_position_limit=False)

    def test_multiple_kill_switches(self, risk_manager):
        """Multiple kill-switches should increment the counter."""
        risk_manager._trigger_kill_switch()
        risk_manager._is_killed = False
        risk_manager._trigger_kill_switch()
        assert risk_manager.kill_count == 2

    def test_kill_switch_skips_cancels_in_read_only_mode(self, state, mock_client):
        """Read-only paper/validation mode must never send mass-cancel commands."""
        risk_manager = RiskManager(
            state=state,
            client=mock_client,
            read_only_mode=True,
            sigma_threshold=3.0,
            cooldown_seconds=5,
            pnl_floor=-0.20,
            max_positions=3,
        )

        risk_manager._trigger_kill_switch()

        assert risk_manager.kill_count == 1
        mock_client.cancel_all_orders.assert_not_called()

    def test_invalidate_private_check_cache_forces_refetch(
        self, risk_manager, mock_client
    ):
        mock_client.get_open_orders.side_effect = [
            [{"id": "1"}],
            [{"id": "1"}, {"id": "2"}, {"id": "3"}],
        ]
        mock_client.get_collateral_balance_allowance.side_effect = [
            SimpleNamespace(balance=10.0, allowance=10.0, available_to_trade=10.0),
            SimpleNamespace(balance=1.0, allowance=0.0, available_to_trade=0.0),
        ]

        assert risk_manager.check_position_limit()
        assert risk_manager.check_balance()

        risk_manager.invalidate_private_check_cache()

        assert not risk_manager.check_position_limit()
        assert not risk_manager.check_balance()
        assert mock_client.get_open_orders.call_count == 2
        assert mock_client.get_collateral_balance_allowance.call_count == 2

    def test_check_volatility_samples_each_state_timestamp_once(
        self, risk_manager, state
    ):
        """Repeated checks without new trades should not double-count the same sample."""
        risk_manager._vol_history.clear()

        base_time = int(time.time() * 1000)
        for i in range(40):
            state.push_trade_sync({
                "price": 95000 + (i * 3),
                "quantity": 0.001,
                "timestamp": base_time + i * 100,
                "is_buyer_maker": False,
                "trade_id": 10_000 + i,
            })

        risk_manager.check_volatility()
        history_len = len(risk_manager._vol_history)

        risk_manager.check_volatility()

        assert len(risk_manager._vol_history) == history_len

    def test_check_volatility_ignores_intra_second_microstructure_bounce(
        self, mock_client
    ):
        """
        The live kill-switch should react to second-level BTC moves, not to
        harmless tick-to-tick bounce within each second.
        """
        state = RollingState(maxlen=10_000)
        risk_manager = RiskManager(
            state=state,
            client=mock_client,
            sigma_threshold=3.0,
            cooldown_seconds=5,
            pnl_floor=-0.20,
            max_positions=3,
        )

        base_time = int(time.time() * 1000)
        trade_id = 0
        triggered = False

        for second in range(120):
            close_price = 95_000 + np.sin(second / 6.0) * 10
            bounce = 8.0 if second < 90 else 11.0

            for offset, price in (
                (0, close_price + bounce),
                (200, close_price - bounce),
                (400, close_price + (bounce * 0.6)),
                (800, close_price),
            ):
                state.push_trade_sync(
                    {
                        "price": float(price),
                        "quantity": 0.001,
                        "timestamp": base_time + (second * 1000) + offset,
                        "is_buyer_maker": False,
                        "trade_id": trade_id,
                    }
                )
                trade_id += 1

            triggered = risk_manager.check_volatility() or triggered

        assert len(risk_manager._vol_history) >= 30
        assert not triggered
        assert risk_manager.kill_count == 0

    def test_check_volatility_ignores_tiny_absolute_spike_even_if_zscore_is_high(
        self, mock_client
    ):
        baseline = [1.0e-5 + ((i % 2) * 1.0e-7) for i in range(30)]
        state = _SequencedVolatilityState(baseline + [3.1e-5])
        risk_manager = RiskManager(
            state=state,
            client=mock_client,
            sigma_threshold=3.0,
            min_absolute_volatility=5.0e-5,
            min_relative_volatility_multiplier=2.0,
            cooldown_seconds=5,
            pnl_floor=-0.20,
            max_positions=3,
        )

        triggered = False
        for _ in range(31):
            state.advance()
            triggered = risk_manager.check_volatility() or triggered

        assert not triggered
        assert risk_manager.kill_count == 0
        mock_client.cancel_all_orders.assert_not_called()

    def test_check_volatility_requires_relative_spike_not_just_zscore(
        self, mock_client
    ):
        baseline = [1.0e-5 + ((i % 2) * 1.0e-7) for i in range(30)]
        state = _SequencedVolatilityState(baseline + [4.2e-5])
        risk_manager = RiskManager(
            state=state,
            client=mock_client,
            sigma_threshold=3.0,
            min_absolute_volatility=2.0e-5,
            min_relative_volatility_multiplier=5.0,
            cooldown_seconds=5,
            pnl_floor=-0.20,
            max_positions=3,
        )

        triggered = False
        for _ in range(31):
            state.advance()
            triggered = risk_manager.check_volatility() or triggered

        assert not triggered
        assert risk_manager.kill_count == 0
        mock_client.cancel_all_orders.assert_not_called()

    def test_check_volatility_triggers_for_large_absolute_and_relative_spike(
        self, mock_client
    ):
        baseline = [1.0e-5 + ((i % 2) * 1.0e-7) for i in range(30)]
        state = _SequencedVolatilityState(baseline + [8.0e-5])
        risk_manager = RiskManager(
            state=state,
            client=mock_client,
            sigma_threshold=3.0,
            min_absolute_volatility=5.0e-5,
            min_relative_volatility_multiplier=4.0,
            cooldown_seconds=5,
            pnl_floor=-0.20,
            max_positions=3,
        )

        triggered = False
        for _ in range(31):
            state.advance()
            triggered = risk_manager.check_volatility() or triggered

        assert triggered
        assert risk_manager.kill_count == 1
        mock_client.cancel_all_orders.assert_called_once()
