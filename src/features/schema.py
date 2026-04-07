"""
Shared feature schema for offline training and live inference.

Keeping this list in one module prevents silent drift between the
training scripts and the real-time execution pipeline.
"""

FEATURE_COLUMNS = (
    "obi",
    "momentum_1m",
    "momentum_2m",
    "momentum_5m",
    "momentum_10m",
    "hurst",
    "fracdiff_close",
    "vol_1m",
    "vol_5m",
    "vol_30m",
    "vol_60m",
    "vwap_dev_5m",
    "vwap_dev_30m",
    "trade_flow_imb",
    "trade_flow_imb_5m",
    "trade_flow_imb_30m",
    "hl_range",
    "vol_ratio_5m",
    "vol_ratio_30m",
    "trades_intensity",
    "trend_alignment",
)

TARGET_COLUMN = "target"
TIMESTAMP_COLUMN = "open_time"


def get_feature_columns() -> list[str]:
    """Return a copy of the model feature columns."""
    return list(FEATURE_COLUMNS)
