"""
Shared minute-bar feature engineering used by both training and live inference.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from config.settings import FEATURES
from src.features.indicators import (
    fractional_differentiation,
    micro_price_momentum,
    order_book_imbalance,
    rolling_hurst_exponent,
    rolling_volatility_fast,
    vwap_deviation_fast,
)

BAR_INPUT_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_buy_base",
    "trades_count",
]


def compute_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the training/live feature frame from 1-minute BTC bars.
    """
    if df.empty:
        return df.copy()

    df = df.sort_values("open_time").reset_index(drop=True).copy()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"] = df["close"].pct_change()

    bid_vol = df["taker_buy_base"].to_numpy(dtype=np.float64)
    ask_vol = (df["volume"] - df["taker_buy_base"]).to_numpy(dtype=np.float64)
    df["obi"] = order_book_imbalance(bid_vol, ask_vol)

    close = df["close"].to_numpy(dtype=np.float64)
    df["momentum_1m"] = micro_price_momentum(close, window=1)
    df["momentum_2m"] = micro_price_momentum(close, window=2)
    df["momentum_5m"] = micro_price_momentum(close, window=5)
    df["momentum_10m"] = micro_price_momentum(close, window=10)

    log_rets = df["log_return"].to_numpy(dtype=np.float64)
    df["hurst"] = rolling_hurst_exponent(
        log_rets,
        window=FEATURES.hurst_window,
        max_lag=FEATURES.hurst_max_lag,
    )

    df["fracdiff_close"] = fractional_differentiation(
        close,
        d=FEATURES.fracdiff_d,
    )

    df["vol_1m"] = rolling_volatility_fast(df["log_return"], window=5)  # Bug 3 fix: was window=1 (always 0)
    df["vol_5m"] = rolling_volatility_fast(df["log_return"], window=5)
    df["vol_30m"] = rolling_volatility_fast(df["log_return"], window=30)
    df["vol_60m"] = rolling_volatility_fast(df["log_return"], window=60)

    df["vwap_dev_5m"] = vwap_deviation_fast(df["close"], df["volume"], window=5)
    df["vwap_dev_30m"] = vwap_deviation_fast(df["close"], df["volume"], window=30)

    total_vol = df["volume"]
    buy_ratio = df["taker_buy_base"] / total_vol.replace(0, np.nan)
    df["trade_flow_imb"] = 2 * buy_ratio - 1
    df["trade_flow_imb_5m"] = (
        df["trade_flow_imb"].rolling(window=5, min_periods=1).mean()
    )
    df["trade_flow_imb_30m"] = (
        df["trade_flow_imb"].rolling(window=30, min_periods=1).mean()
    )

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["vol_ratio_5m"] = (
        df["volume"] / df["volume"].rolling(window=5, min_periods=1).mean()
    )
    df["vol_ratio_30m"] = (
        df["volume"] / df["volume"].rolling(window=30, min_periods=1).mean()
    )
    df["trades_intensity"] = (
        df["trades_count"]
        / df["trades_count"].rolling(window=30, min_periods=1).mean()
    )

    # Improvement 8: Multi-timeframe trend alignment
    # +3/-3 = all timeframes agree, 0 = mixed signals
    df["trend_alignment"] = (
        np.sign(df["momentum_1m"].fillna(0))
        + np.sign(df["momentum_5m"].fillna(0))
        + np.sign(df["momentum_10m"].fillna(0))
    )

    return df


def aggregate_trades_to_1m_bars(
    trades: Iterable[object],
    include_incomplete_last_bar: bool = False,
) -> pd.DataFrame:
    """
    Aggregate tick trades into synthetic 1-minute bars.

    The live model was trained on minute bars, so we drop the trailing partial
    minute by default and only emit fully-formed bars.
    """
    rows = [
        {
            "timestamp": int(trade.timestamp),
            "price": float(trade.price),
            "quantity": float(trade.quantity),
            "is_buyer_maker": bool(trade.is_buyer_maker),
        }
        for trade in trades
    ]
    if not rows:
        return pd.DataFrame(columns=BAR_INPUT_COLUMNS)

    df = pd.DataFrame.from_records(rows)
    df["bucket_start_ms"] = (df["timestamp"] // 60000) * 60000

    latest_bucket_start = int(df["bucket_start_ms"].iloc[-1])
    latest_bucket_age_ms = int(df["timestamp"].iloc[-1]) - latest_bucket_start
    if not include_incomplete_last_bar and latest_bucket_age_ms < 59000:
        df = df[df["bucket_start_ms"] < latest_bucket_start]

    if df.empty:
        return pd.DataFrame(columns=BAR_INPUT_COLUMNS)

    df["taker_buy_base"] = np.where(~df["is_buyer_maker"], df["quantity"], 0.0)
    bars = (
        df.groupby("bucket_start_ms", sort=True)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("quantity", "sum"),
            taker_buy_base=("taker_buy_base", "sum"),
            trades_count=("price", "size"),
        )
        .reset_index()
    )
    bars.insert(0, "open_time", pd.to_datetime(bars["bucket_start_ms"], unit="ms"))
    bars = bars.drop(columns=["bucket_start_ms"])
    return bars[BAR_INPUT_COLUMNS]
