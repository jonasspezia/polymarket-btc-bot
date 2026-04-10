from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.utils.run_governance import (
    RunManifestManager,
    RuntimeConfigurationError,
    build_runtime_config_snapshot,
    validate_runtime_configuration,
)


@dataclass(frozen=True)
class _FakePolymarketConfig:
    clob_host: str = "https://clob.polymarket.com"
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    data_api_base: str = "https://data-api.polymarket.com"
    event_base: str = "https://polymarket.com/event"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    chain_id: int = 137
    private_key: str = "super-secret-private-key"
    api_key: str = "api-key-1234"
    api_secret: str = "api-secret-1234"
    api_passphrase: str = "api-passphrase-1234"
    funder_address: str = "0x1234567890abcdef"
    signature_type: int = 2
    event_slug: str = "bitcoin-above-on-april-6-2026-2pm-et"
    market_poll_interval_seconds: int = 240
    btc_market_keywords: tuple = ("btc", "bitcoin")


@dataclass(frozen=True)
class _FakeTradingConfig:
    dry_run: bool = False
    validation_only_mode: bool = False
    live_trading_enabled: bool = True
    allow_non_5m_live_markets: bool = False
    order_size: float = 1.0
    order_notional: float = 0.0
    min_edge: float = 0.04
    min_side_probability: float = 0.56
    max_entry_price: float = 0.68
    max_spread: float = 0.12
    strategy_style: str = "momentum"
    probability_vol_window_seconds: int = 300
    probability_min_sigma: float = 0.0015
    order_book_imbalance_levels: int = 5
    min_order_book_imbalance: float = 0.35
    max_ask_wall_ratio: float = 2.5
    gtd_ttl_seconds: int = 10
    post_only: bool = True
    max_open_positions: int = 1
    duplicate_signal_window_seconds: int = 15
    require_live_test_before_live_orders: bool = True
    live_test_window_seconds: int = 600
    live_test_min_completed_markets: int = 2
    live_test_min_win_rate: float = 0.5
    live_test_min_profit: float = 0.01
    live_test_max_cumulative_loss: float = 0.0
    allow_upsize_to_min_order_size: bool = False
    bankroll_fraction_per_order: float = 0.25
    max_order_notional: float = 0.0
    reserve_collateral_amount: float = 0.0
    enable_dynamic_exits: bool = True
    stop_loss_factor: float = 0.5
    take_profit_multiple: float = 3.0
    take_profit_fraction: float = 0.5
    time_decay_exit_seconds: int = 180
    time_decay_distance_pct: float = 0.005
    min_time_remaining_seconds: int = 120
    use_kelly_sizing: bool = False
    kelly_fraction: float = 0.10
    tick_size: str = "0.01"


@dataclass(frozen=True)
class _FakeRiskConfig:
    volatility_sigma_threshold: float = 3.0
    volatility_min_absolute_threshold: float = 0.00005
    volatility_min_relative_multiplier: float = 4.0
    kill_switch_cooldown_seconds: int = 60
    pnl_floor: float = -0.2
    min_available_collateral: float = 10.0
    max_available_collateral_drawdown: float = 1.0
    private_check_cache_ttl_seconds: float = 5.0
    vol_lookback_window: int = 300


@dataclass(frozen=True)
class _FakeBinanceConfig:
    ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
    rest_base: str = "https://api.binance.com"
    kline_endpoint: str = "/api/v3/klines"
    agg_trades_endpoint: str = "/api/v3/aggTrades"
    symbol: str = "BTCUSDT"
    reconnect_delay_base: float = 1.0
    reconnect_delay_max: float = 60.0
    connection_lifetime_hours: int = 23


def test_validate_runtime_configuration_rejects_inconsistent_notional_limits(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.utils.run_governance.TRADING",
        SimpleNamespace(
            live_trading_enabled=True,
            min_edge=0.04,
            min_side_probability=0.56,
            max_entry_price=0.68,
            max_spread=0.12,
            gtd_ttl_seconds=10,
            max_open_positions=1,
            duplicate_signal_window_seconds=15,
            order_size=1.0,
            order_notional=5.0,
            max_order_notional=1.0,
            bankroll_fraction_per_order=0.25,
            reserve_collateral_amount=0.0,
            min_time_remaining_seconds=120,
            time_decay_exit_seconds=180,
        ),
    )
    monkeypatch.setattr(
        "src.utils.run_governance.RISK",
        SimpleNamespace(
            volatility_sigma_threshold=3.0,
            volatility_min_absolute_threshold=0.00005,
            volatility_min_relative_multiplier=4.0,
            kill_switch_cooldown_seconds=60,
            private_check_cache_ttl_seconds=5.0,
            pnl_floor=-0.2,
            min_available_collateral=10.0,
            max_available_collateral_drawdown=1.0,
        ),
    )

    with pytest.raises(RuntimeConfigurationError, match="ORDER_NOTIONAL"):
        validate_runtime_configuration(dry_run=False, validation_only=False)


def test_validate_runtime_configuration_rejects_invalid_volatility_guards(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.utils.run_governance.TRADING",
        SimpleNamespace(
            live_trading_enabled=True,
            min_edge=0.04,
            min_side_probability=0.56,
            max_entry_price=0.68,
            max_spread=0.12,
            gtd_ttl_seconds=10,
            max_open_positions=1,
            duplicate_signal_window_seconds=15,
            order_size=1.0,
            order_notional=0.0,
            max_order_notional=0.0,
            bankroll_fraction_per_order=0.25,
            reserve_collateral_amount=0.0,
            min_time_remaining_seconds=120,
            time_decay_exit_seconds=180,
        ),
    )
    monkeypatch.setattr(
        "src.utils.run_governance.RISK",
        SimpleNamespace(
            volatility_sigma_threshold=3.0,
            volatility_min_absolute_threshold=-0.1,
            volatility_min_relative_multiplier=0.5,
            kill_switch_cooldown_seconds=60,
            private_check_cache_ttl_seconds=5.0,
            pnl_floor=-0.2,
            min_available_collateral=10.0,
            max_available_collateral_drawdown=1.0,
        ),
    )

    with pytest.raises(
        RuntimeConfigurationError,
        match="VOLATILITY_MIN_ABSOLUTE_THRESHOLD",
    ):
        validate_runtime_configuration(dry_run=False, validation_only=False)


def test_validate_runtime_configuration_rejects_permissive_live_settings(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.utils.run_governance.TRADING",
        SimpleNamespace(
            live_trading_enabled=True,
            min_edge=0.02,
            min_side_probability=0.52,
            max_entry_price=0.90,
            max_spread=0.30,
            gtd_ttl_seconds=10,
            max_open_positions=3,
            duplicate_signal_window_seconds=15,
            order_size=1.0,
            order_notional=0.0,
            max_order_notional=0.0,
            bankroll_fraction_per_order=1.0,
            reserve_collateral_amount=0.0,
            min_time_remaining_seconds=30,
            time_decay_exit_seconds=1800,
        ),
    )
    monkeypatch.setattr(
        "src.utils.run_governance.RISK",
        SimpleNamespace(
            volatility_sigma_threshold=3.0,
            volatility_min_absolute_threshold=0.00005,
            volatility_min_relative_multiplier=4.0,
            kill_switch_cooldown_seconds=60,
            private_check_cache_ttl_seconds=5.0,
            pnl_floor=-0.2,
            min_available_collateral=0.0,
            max_available_collateral_drawdown=0.0,
        ),
    )

    with pytest.raises(
        RuntimeConfigurationError,
        match="MIN_SIDE_PROBABILITY|MAX_ENTRY_PRICE|MAX_SPREAD|MAX_OPEN_POSITIONS",
    ):
        validate_runtime_configuration(dry_run=False, validation_only=False)


def test_build_runtime_config_snapshot_redacts_secrets(monkeypatch):
    monkeypatch.setattr(
        "src.utils.run_governance.POLYMARKET",
        _FakePolymarketConfig(),
    )
    monkeypatch.setattr(
        "src.utils.run_governance.TRADING",
        _FakeTradingConfig(),
    )
    monkeypatch.setattr(
        "src.utils.run_governance.RISK",
        _FakeRiskConfig(),
    )
    monkeypatch.setattr(
        "src.utils.run_governance.BINANCE",
        _FakeBinanceConfig(),
    )
    monkeypatch.setattr(
        "src.utils.run_governance.PATHS",
        SimpleNamespace(
            model_path="data/models/lgbm_btc_5m.txt",
            artifacts_dir="data/artifacts",
            run_manifests_dir="data/artifacts/runs",
        ),
    )

    snapshot = build_runtime_config_snapshot(dry_run=False, validation_only=False)

    assert snapshot["effective_mode"]["live"] is True
    assert snapshot["polymarket"]["private_key"] != "super-secret-private-key"
    assert snapshot["polymarket"]["api_secret"] != "api-secret-1234"
    assert snapshot["polymarket"]["has_private_key"] is True
    assert snapshot["polymarket"]["has_api_credentials"] is True
    assert snapshot["paths"]["model_target_horizon_minutes"] == 5


def test_run_manifest_manager_writes_start_and_finalize_artifact(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        RunManifestManager,
        "_build_run_id",
        staticmethod(lambda: "20260406T120000Z-test1234"),
    )
    monkeypatch.setattr(
        RunManifestManager,
        "_get_git_metadata",
        staticmethod(
            lambda: {
                "branch": "main",
                "commit": "abc123",
                "is_dirty": False,
            }
        ),
    )

    manager = RunManifestManager(manifests_dir=str(tmp_path))
    manager.start(
        mode_label="LIVE",
        config_snapshot={"effective_mode": {"live": True}},
    )
    manager.mark_running(runtime_summary={"orders_placed": 0})
    manager.finalize(
        status="shutdown_complete",
        runtime_summary={"orders_placed": 1},
    )

    manifest_path = tmp_path / "20260406T120000Z-test1234.json"
    assert manifest_path.exists()

    manifest = manifest_path.read_text(encoding="utf-8")
    assert '"status": "shutdown_complete"' in manifest
    assert '"run_id": "20260406T120000Z-test1234"' in manifest
    assert '"orders_placed": 1' in manifest
