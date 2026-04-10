"""
Centralized configuration for the Polymarket BTC trading bot.
All values are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a safe default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BinanceConfig:
    """Binance WebSocket and REST API configuration."""
    ws_url: str = "wss://fstream.binance.com/stream?streams=btcusdt@aggTrade/btcusdt@forceOrder"
    ws_url_spot_fallback: str = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"  # Spot fallback (no liquidations)
    rest_base: str = "https://api.binance.com"
    kline_endpoint: str = "/api/v3/klines"
    agg_trades_endpoint: str = "/api/v3/aggTrades"
    symbol: str = "BTCUSDT"
    reconnect_delay_base: float = 1.0
    reconnect_delay_max: float = 60.0
    connection_lifetime_hours: int = 23  # Rotate before 24h Binance limit
    enable_spot_fallback: bool = field(
        default_factory=lambda: _get_bool_env("BINANCE_ENABLE_SPOT_FALLBACK", True)
    )  # Enable automatic fallback to Spot if Futures fails


@dataclass(frozen=True)
class PolymarketConfig:
    """Polymarket CLOB and Gamma API configuration."""
    clob_host: str = "https://clob.polymarket.com"
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    data_api_base: str = "https://data-api.polymarket.com"
    event_base: str = "https://polymarket.com/event"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    chain_id: int = 137  # Polygon mainnet

    # Authentication (loaded from env)
    private_key: str = field(default_factory=lambda: os.getenv("POLYGON_PRIVATE_KEY", ""))
    api_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_SECRET", ""))
    api_passphrase: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_PASSPHRASE", ""))
    funder_address: str = field(default_factory=lambda: os.getenv("FUNDER_ADDRESS", ""))
    signature_type: int = field(
        default_factory=lambda: int(os.getenv("SIGNATURE_TYPE", "0"))
    )
    event_slug: str = field(default_factory=lambda: os.getenv("POLYMARKET_EVENT_SLUG", ""))

    # Market discovery
    market_poll_interval_seconds: int = 240  # Refresh every 4 min (before 5-min resolution)
    market_poll_retry_seconds: int = field(
        default_factory=lambda: int(os.getenv("MARKET_POLL_RETRY_SECONDS", "5"))
    )
    market_rejection_backoff_seconds: int = field(
        default_factory=lambda: int(os.getenv("MARKET_REJECTION_BACKOFF_SECONDS", "15"))
    )
    market_pathological_backoff_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("MARKET_PATHOLOGICAL_BACKOFF_SECONDS", "60")
        )
    )
    btc_market_keywords: tuple = ("btc", "bitcoin", "5 min", "5-min", "5min")


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering parameters."""
    # Rolling state buffer
    rolling_state_maxlen: int = field(
        default_factory=lambda: int(os.getenv("ROLLING_STATE_MAXLEN", "50000"))
    )
    minimum_warmup_trades: int = field(
        default_factory=lambda: int(os.getenv("MIN_WARMUP_TRADES", "1000"))
    )
    minimum_history_seconds: int = field(
        default_factory=lambda: int(os.getenv("MIN_HISTORY_SECONDS", "3600"))
    )
    
    # Momentum windows (in seconds)
    momentum_window_30s: int = 30
    momentum_window_60s: int = 60
    
    # Hurst exponent
    hurst_window: int = 100
    hurst_max_lag: int = 20
    
    # Volatility windows (in seconds)
    vol_window_30s: int = 30
    vol_window_60s: int = 60
    vol_window_300s: int = 300
    
    # VWAP
    vwap_window_seconds: int = 300  # 5 minutes
    
    # Order book imbalance
    obi_window: int = 60  # ticks
    
    # Fractional differentiation
    fracdiff_d: float = 0.5  # Starting value; optimized via ADF test during training


@dataclass(frozen=True)
class TradingConfig:
    """Trading execution parameters."""
    dry_run: bool = field(
        default_factory=lambda: _get_bool_env("DRY_RUN", True)
    )
    validation_only_mode: bool = field(
        default_factory=lambda: _get_bool_env("VALIDATION_ONLY_MODE", False)
    )
    live_trading_enabled: bool = field(
        default_factory=lambda: _get_bool_env("LIVE_TRADING_ENABLED", False)
    )
    allow_non_5m_live_markets: bool = field(
        default_factory=lambda: _get_bool_env("ALLOW_NON_5M_LIVE_MARKETS", False)
    )
    order_size: float = field(
        default_factory=lambda: float(os.getenv("ORDER_SIZE", "1.0"))
    )
    order_notional: float = field(
        default_factory=lambda: float(os.getenv("ORDER_NOTIONAL", "0.0"))
    )
    min_edge: float = field(
        default_factory=lambda: float(os.getenv("MIN_EDGE", "0.04"))
    )
    min_side_probability: float = field(
        default_factory=lambda: float(os.getenv("MIN_SIDE_PROBABILITY", "0.56"))
    )
    max_entry_price: float = field(
        default_factory=lambda: float(os.getenv("MAX_ENTRY_PRICE", "0.68"))
    )
    max_spread: float = field(
        default_factory=lambda: float(os.getenv("MAX_SPREAD", "0.12"))
    )
    strategy_style: str = field(
        default_factory=lambda: os.getenv("STRATEGY_STYLE", "momentum").strip().lower()
    )
    probability_vol_window_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("PROBABILITY_VOL_WINDOW_SECONDS", "300")
        )
    )
    probability_min_sigma: float = field(
        default_factory=lambda: float(os.getenv("PROBABILITY_MIN_SIGMA", "0.0015"))
    )
    order_book_imbalance_levels: int = field(
        default_factory=lambda: int(
            os.getenv("ORDER_BOOK_IMBALANCE_LEVELS", "5")
        )
    )
    min_order_book_imbalance: float = field(
        default_factory=lambda: float(
            os.getenv("MIN_ORDER_BOOK_IMBALANCE", "0.35")
        )
    )
    max_ask_wall_ratio: float = field(
        default_factory=lambda: float(os.getenv("MAX_ASK_WALL_RATIO", "2.5"))
    )
    gtd_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("GTD_TTL_SECONDS", "10"))
    )
    post_only: bool = True  # MANDATORY — never change this
    max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "1"))
    )
    duplicate_signal_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("DUPLICATE_SIGNAL_WINDOW_SECONDS", "15"))
    )
    require_live_test_before_live_orders: bool = field(
        default_factory=lambda: _get_bool_env(
            "REQUIRE_LIVE_TEST_BEFORE_LIVE_ORDERS", True
        )
    )
    live_test_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("LIVE_TEST_WINDOW_SECONDS", "600"))
    )
    live_test_min_completed_markets: int = field(
        default_factory=lambda: int(
            os.getenv("LIVE_TEST_MIN_COMPLETED_MARKETS", "2")
        )
    )
    live_test_min_win_rate: float = field(
        default_factory=lambda: float(os.getenv("LIVE_TEST_MIN_WIN_RATE", "0.50"))
    )
    live_test_min_profit: float = field(
        default_factory=lambda: float(os.getenv("LIVE_TEST_MIN_PROFIT", "0.01"))
    )
    live_test_max_cumulative_loss: float = field(
        default_factory=lambda: float(
            os.getenv("LIVE_TEST_MAX_CUMULATIVE_LOSS", "0.00")
        )
    )
    allow_upsize_to_min_order_size: bool = field(
        default_factory=lambda: _get_bool_env(
            "ALLOW_UPSIZE_TO_MIN_ORDER_SIZE", False
        )
    )
    bankroll_fraction_per_order: float = field(
        default_factory=lambda: float(
            os.getenv("BANKROLL_FRACTION_PER_ORDER", "0.25")
        )
    )
    max_order_notional: float = field(
        default_factory=lambda: float(os.getenv("MAX_ORDER_NOTIONAL", "0.0"))
    )
    reserve_collateral_amount: float = field(
        default_factory=lambda: float(
            os.getenv("RESERVE_COLLATERAL_AMOUNT", "0.0")
        )
    )
    enable_dynamic_exits: bool = field(
        default_factory=lambda: _get_bool_env("ENABLE_DYNAMIC_EXITS", True)
    )
    stop_loss_factor: float = field(
        default_factory=lambda: float(os.getenv("STOP_LOSS_FACTOR", "0.5"))
    )  # Fraction of entry price you're willing to lose (0.5 = stop at entry * 0.5)
    take_profit_multiple: float = field(
        default_factory=lambda: float(os.getenv("TAKE_PROFIT_MULTIPLE", "0.3"))
    )  # Fraction of remaining edge to capture (0.3 = TP at entry + 0.3*(1-entry))
    take_profit_fraction: float = field(
        default_factory=lambda: float(os.getenv("TAKE_PROFIT_FRACTION", "0.5"))
    )
    time_decay_exit_seconds: int = field(
        default_factory=lambda: int(os.getenv("TIME_DECAY_EXIT_SECONDS", "180"))
    )
    time_decay_distance_pct: float = field(
        default_factory=lambda: float(
            os.getenv("TIME_DECAY_DISTANCE_PCT", "0.005")
        )
    )
    min_time_remaining_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("MIN_TIME_REMAINING_SECONDS", "120")
        )
    )
    use_kelly_sizing: bool = field(
        default_factory=lambda: _get_bool_env("USE_KELLY_SIZING", False)
    )
    kelly_fraction: float = field(
        default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.10"))
    )
    tick_size: str = "0.01"  # Polymarket tick size


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""
    volatility_sigma_threshold: float = field(
        default_factory=lambda: float(os.getenv("VOLATILITY_SIGMA_THRESHOLD", "3.0"))
    )
    volatility_min_absolute_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("VOLATILITY_MIN_ABSOLUTE_THRESHOLD", "0.00005")
        )
    )
    volatility_min_relative_multiplier: float = field(
        default_factory=lambda: float(
            os.getenv("VOLATILITY_MIN_RELATIVE_MULTIPLIER", "4.0")
        )
    )
    kill_switch_cooldown_seconds: int = field(
        default_factory=lambda: int(os.getenv("KILL_SWITCH_COOLDOWN_SECONDS", "60"))
    )
    pnl_floor: float = field(
        default_factory=lambda: float(os.getenv("PNL_FLOOR", "-0.20"))
    )
    min_available_collateral: float = field(
        default_factory=lambda: float(
            os.getenv("MIN_AVAILABLE_COLLATERAL", "10.0")
        )
    )
    max_available_collateral_drawdown: float = field(
        default_factory=lambda: float(
            os.getenv("MAX_AVAILABLE_COLLATERAL_DRAWDOWN", "1.0")
        )
    )
    private_check_cache_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("PRIVATE_CHECK_CACHE_TTL_SECONDS", "5.0")
        )
    )
    vol_lookback_window: int = 300  # seconds for σ baseline calculation


@dataclass(frozen=True)
class PathConfig:
    """File system paths."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "data/models"
    artifacts_dir: str = field(
        default_factory=lambda: os.getenv("ARTIFACTS_DIR", "data/artifacts")
    )
    model_filename: str = "lgbm_btc_5m.txt"
    features_filename: str = "features.parquet"
    klines_filename: str = "btc_1m_klines.csv"
    agg_trades_filename: str = "btc_agg_trades.csv"

    @property
    def model_path(self) -> str:
        return os.path.join(self.models_dir, self.model_filename)

    @property
    def features_path(self) -> str:
        return os.path.join(self.processed_data_dir, self.features_filename)

    @property
    def klines_path(self) -> str:
        return os.path.join(self.raw_data_dir, self.klines_filename)

    @property
    def agg_trades_path(self) -> str:
        return os.path.join(self.raw_data_dir, self.agg_trades_filename)

    @property
    def run_manifests_dir(self) -> str:
        return os.path.join(self.artifacts_dir, "runs")

    @property
    def experiments_dir(self) -> str:
        return os.path.join(self.artifacts_dir, "experiments")


# ---------------------------------------------------------------------------
# Singleton instances — import these directly
# ---------------------------------------------------------------------------
BINANCE = BinanceConfig()
POLYMARKET = PolymarketConfig()
FEATURES = FeatureConfig()
TRADING = TradingConfig()
RISK = RiskConfig()
PATHS = PathConfig()
