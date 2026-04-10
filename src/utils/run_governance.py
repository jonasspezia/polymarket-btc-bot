"""
Runtime governance helpers for institutional-style operating discipline.

This module provides:
1. Fail-closed validation for critical runtime settings
2. Persistent run manifests for reproducibility and auditability
"""

import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import BINANCE, PATHS, POLYMARKET, RISK, TRADING
from src.utils.model_metadata import get_model_target_horizon_minutes


class RuntimeConfigurationError(ValueError):
    """Raised when runtime settings are unsafe or internally inconsistent."""


def validate_runtime_configuration(
    *,
    dry_run: bool,
    validation_only: bool,
) -> None:
    """Fail closed on runtime settings that are unsafe or incoherent."""
    errors: list[str] = []
    live_mode = not dry_run and not validation_only

    if live_mode and not TRADING.live_trading_enabled:
        errors.append("LIVE_TRADING_ENABLED must be true for live mode")

    if TRADING.min_edge < 0:
        errors.append("MIN_EDGE must be non-negative")
    if not 0 < TRADING.min_side_probability < 1:
        errors.append("MIN_SIDE_PROBABILITY must be between 0 and 1")
    if not 0 < TRADING.max_entry_price <= 1:
        errors.append("MAX_ENTRY_PRICE must be in the interval (0, 1]")
    if TRADING.gtd_ttl_seconds <= 0:
        errors.append("GTD_TTL_SECONDS must be positive")
    if TRADING.max_open_positions < 1:
        errors.append("MAX_OPEN_POSITIONS must be at least 1")
    if TRADING.duplicate_signal_window_seconds < 0:
        errors.append("DUPLICATE_SIGNAL_WINDOW_SECONDS must be non-negative")
    if TRADING.order_size <= 0 and TRADING.order_notional <= 0:
        errors.append("ORDER_SIZE or ORDER_NOTIONAL must be positive")
    if TRADING.order_notional < 0:
        errors.append("ORDER_NOTIONAL must be non-negative")
    if TRADING.max_order_notional < 0:
        errors.append("MAX_ORDER_NOTIONAL must be non-negative")
    if (
        TRADING.order_notional > 0
        and TRADING.max_order_notional > 0
        and TRADING.order_notional > TRADING.max_order_notional + 1e-9
    ):
        errors.append("ORDER_NOTIONAL cannot exceed MAX_ORDER_NOTIONAL")
    if not 0 < TRADING.bankroll_fraction_per_order <= 1:
        errors.append("BANKROLL_FRACTION_PER_ORDER must be in the interval (0, 1]")
    if TRADING.reserve_collateral_amount < 0:
        errors.append("RESERVE_COLLATERAL_AMOUNT must be non-negative")

    if RISK.volatility_sigma_threshold <= 0:
        errors.append("VOLATILITY_SIGMA_THRESHOLD must be positive")
    if RISK.volatility_min_absolute_threshold < 0:
        errors.append("VOLATILITY_MIN_ABSOLUTE_THRESHOLD must be non-negative")
    if RISK.volatility_min_relative_multiplier < 1:
        errors.append("VOLATILITY_MIN_RELATIVE_MULTIPLIER must be at least 1")
    if RISK.kill_switch_cooldown_seconds < 0:
        errors.append("KILL_SWITCH_COOLDOWN_SECONDS must be non-negative")
    if RISK.private_check_cache_ttl_seconds < 0:
        errors.append("PRIVATE_CHECK_CACHE_TTL_SECONDS must be non-negative")
    if RISK.pnl_floor > 0:
        errors.append("PNL_FLOOR must be zero or negative")

    if live_mode:
        max_spread = float(getattr(TRADING, "max_spread", 0.30))
        min_time_remaining_seconds = int(
            getattr(TRADING, "min_time_remaining_seconds", 60)
        )
        time_decay_exit_seconds = int(
            getattr(TRADING, "time_decay_exit_seconds", 1800)
        )
        min_available_collateral = float(
            getattr(RISK, "min_available_collateral", 0.0)
        )
        max_available_collateral_drawdown = float(
            getattr(RISK, "max_available_collateral_drawdown", 0.0)
        )

        if TRADING.min_edge < 0.03:
            errors.append("MIN_EDGE must be at least 0.03 in live mode")
        if TRADING.min_side_probability < 0.55:
            errors.append(
                "MIN_SIDE_PROBABILITY must be at least 0.55 in live mode"
            )
        if TRADING.max_entry_price > 0.70:
            errors.append("MAX_ENTRY_PRICE must be at most 0.70 in live mode")
        if max_spread > 0.20:
            errors.append("MAX_SPREAD must be at most 0.20 in live mode")
        if TRADING.max_open_positions > 1:
            errors.append("MAX_OPEN_POSITIONS must be 1 in live mode")
        if TRADING.bankroll_fraction_per_order > 0.25:
            errors.append(
                "BANKROLL_FRACTION_PER_ORDER must be at most 0.25 in live mode"
            )
        if min_time_remaining_seconds < 60:
            errors.append(
                "MIN_TIME_REMAINING_SECONDS must be at least 60 in live mode"
            )
        if time_decay_exit_seconds > 300:
            errors.append(
                "TIME_DECAY_EXIT_SECONDS must be at most 300 in live mode"
            )
        if min_available_collateral < 10:
            errors.append(
                "MIN_AVAILABLE_COLLATERAL must be at least 10 in live mode"
            )
        if max_available_collateral_drawdown <= 0:
            errors.append(
                "MAX_AVAILABLE_COLLATERAL_DRAWDOWN must be positive in live mode"
            )
        elif max_available_collateral_drawdown > 5:
            errors.append(
                "MAX_AVAILABLE_COLLATERAL_DRAWDOWN must be at most 5 in live mode"
            )

    if errors:
        raise RuntimeConfigurationError("; ".join(errors))


def build_runtime_config_snapshot(
    *,
    dry_run: bool,
    validation_only: bool,
) -> dict[str, Any]:
    """Return a sanitized runtime snapshot suitable for audit artifacts."""
    polymarket = asdict(POLYMARKET)
    for secret_key in (
        "private_key",
        "api_key",
        "api_secret",
        "api_passphrase",
        "funder_address",
    ):
        polymarket[secret_key] = _redact_sensitive_value(polymarket.get(secret_key))

    polymarket["has_private_key"] = bool(POLYMARKET.private_key)
    polymarket["has_api_credentials"] = bool(
        POLYMARKET.api_key and POLYMARKET.api_secret and POLYMARKET.api_passphrase
    )

    return {
        "effective_mode": {
            "dry_run": dry_run,
            "validation_only": validation_only,
            "live": not dry_run and not validation_only,
        },
        "trading": asdict(TRADING),
        "risk": asdict(RISK),
        "binance": asdict(BINANCE),
        "polymarket": polymarket,
        "paths": {
            "model_path": PATHS.model_path,
            "model_target_horizon_minutes": get_model_target_horizon_minutes(
                PATHS.model_path
            ),
            "artifacts_dir": PATHS.artifacts_dir,
            "run_manifests_dir": PATHS.run_manifests_dir,
        },
    }


class RunManifestManager:
    """Write an immutable per-run manifest for audit and reproducibility."""

    def __init__(self, manifests_dir: Optional[str] = None):
        self._manifests_dir = Path(manifests_dir or PATHS.run_manifests_dir)
        self._run_id = self._build_run_id()
        self._manifest_path = self._manifests_dir / f"{self._run_id}.json"
        self._start_monotonic = time.monotonic()
        self._manifest: dict[str, Any] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def manifest_path(self) -> str:
        return str(self._manifest_path)

    def start(self, *, mode_label: str, config_snapshot: dict[str, Any]) -> None:
        """Initialize the manifest at process startup."""
        self._manifest = {
            "run_id": self._run_id,
            "status": "starting",
            "mode": mode_label,
            "started_at_utc": _utc_now_iso(),
            "process": {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "cwd": os.getcwd(),
            },
            "git": self._get_git_metadata(),
            "config": config_snapshot,
        }
        self._write_manifest()

    def mark_running(self, runtime_summary: Optional[dict[str, Any]] = None) -> None:
        """Promote the manifest to a running state once startup succeeds."""
        self._manifest["status"] = "running"
        self._manifest["running_at_utc"] = _utc_now_iso()
        if runtime_summary is not None:
            self._manifest["runtime_summary"] = runtime_summary
        self._write_manifest()

    def finalize(
        self,
        *,
        status: str,
        runtime_summary: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Finalize the manifest with exit metadata."""
        self._manifest["status"] = status
        self._manifest["ended_at_utc"] = _utc_now_iso()
        self._manifest["duration_seconds"] = round(
            time.monotonic() - self._start_monotonic,
            3,
        )
        if runtime_summary is not None:
            self._manifest["runtime_summary"] = runtime_summary
        if error:
            self._manifest["last_error"] = error
        self._write_manifest()

    def _write_manifest(self) -> None:
        self._manifests_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self._manifest_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self._manifest_path)

    @staticmethod
    def _build_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _get_git_metadata() -> dict[str, Any]:
        repo_root = Path(__file__).resolve().parents[2]
        return {
            "branch": _safe_git_output(
                ["git", "branch", "--show-current"],
                cwd=repo_root,
            ),
            "commit": _safe_git_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
            ),
            "is_dirty": bool(
                _safe_git_output(
                    ["git", "status", "--porcelain"],
                    cwd=repo_root,
                )
            ),
        }


def _safe_git_output(command: list[str], *, cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    output = completed.stdout.strip()
    return output or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact_sensitive_value(value: object) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value)
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}...{text[-4:]}"
