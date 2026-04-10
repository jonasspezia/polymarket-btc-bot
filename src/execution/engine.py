"""
Main asyncio execution engine.
Orchestrates all concurrent tasks: market data ingestion, market discovery,
real-time inference, order routing, and risk monitoring.

Usage:
    python -m src.execution.engine
"""

from collections import Counter
import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import POLYMARKET, TRADING
from src.exchange.binance_rest import BinanceRESTClient
from src.exchange.binance_ws import BinanceWebSocket
from src.exchange.gamma_api import GammaAPIClient, MarketInfo
from src.exchange.polymarket_client import PolymarketClient
from src.exchange.polymarket_ws import PolymarketWebSocket
from src.execution.inference import ModelInference
from src.execution.live_test_gate import LiveTestGate
from src.execution.order_router import OrderRouter
from src.execution.position_manager import PositionManager
from src.execution.probability_estimator import MarketProbabilityEstimator
from src.execution.risk_manager import RiskManager
from src.features.pipeline import FeaturePipeline
from src.features.trend_filter import TrendFilter
from src.utils.logging_config import setup_logging
from src.utils.model_metadata import DEFAULT_TARGET_HORIZON_MINUTES
from src.utils.run_governance import (
    RunManifestManager,
    RuntimeConfigurationError,
    build_runtime_config_snapshot,
    validate_runtime_configuration,
)
from src.utils.state import RollingState

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine that coordinates all components via asyncio.
    
    Architecture:
        Task 1: Binance WebSocket consumer → feeds RollingState
        Task 2: State ingestion loop → moves data from WS queue to state
        Task 3: Gamma API poller → discovers active 5-min BTC markets
        Task 4: Inference + Trading loop → predict & route orders
        Task 5: Risk monitoring loop → volatility kill-switch
    """

    def __init__(
        self,
        dry_run_override: Optional[bool] = None,
        validation_only_override: Optional[bool] = None,
    ):
        # Core state
        self._state = RollingState()
        self._model: Optional[ModelInference] = None
        self._models: dict[int, ModelInference] = {}
        self._probability_estimator = MarketProbabilityEstimator()
        self._pipeline: Optional[FeaturePipeline] = None
        self._pipelines: dict[int, FeaturePipeline] = {}
        self._trend_filter = TrendFilter()
        
        # Exchange clients
        self._binance_rest = BinanceRESTClient()
        self._binance_ws = BinanceWebSocket()
        self._gamma = GammaAPIClient()
        self._pm_client: Optional[PolymarketClient] = None
        self._pm_ws = PolymarketWebSocket()
        
        # Execution
        self._dry_run = dry_run_override if dry_run_override is not None else TRADING.dry_run
        self._validation_only_mode = (
            validation_only_override
            if validation_only_override is not None
            else bool(getattr(TRADING, "validation_only_mode", False))
        )
        self._router: Optional[OrderRouter] = None
        self._risk: Optional[RiskManager] = None
        self._live_test_gate: Optional[LiveTestGate] = None
        self._position_manager: Optional[PositionManager] = None
        
        # Current market
        self._active_market: Optional[MarketInfo] = None
        self._market_ready_event = asyncio.Event()
        self._inference_waiting_for_market_logged = False
        self._near_expiry_refresh_cooldown_seconds = 5.0
        self._next_near_expiry_refresh_at = 0.0
        self._near_expiry_refresh_condition_id: Optional[str] = None
        self._market_rejection_backoff_until_by_key: dict[str, float] = {}
        self._market_family_rejection_backoff_until_by_key: dict[str, float] = {}
        self._tracked_realized_pnl_by_position: dict[str, float] = {}
        self._realized_pnl_baseline_initialized = False
        self._last_live_horizon_notice_key: Optional[tuple[str, str, Optional[int]]] = None
        self._run_manifest_manager: Optional[RunManifestManager] = None
        
        # Control
        self._running = False
        self._shutdown_event = asyncio.Event()

    def _is_validation_only_mode(self) -> bool:
        """Return True when authenticated checks should run without live orders."""
        return self._validation_only_mode

    def _set_active_market(self, market: Optional[MarketInfo]) -> None:
        """Keep active-market state and inference gating in sync."""
        self._active_market = market
        if market is None:
            self._market_ready_event.clear()
            return

        self._market_ready_event.set()
        self._inference_waiting_for_market_logged = False

    def _mode_label(self) -> str:
        if self._is_validation_only_mode():
            return "VALIDATION_ONLY"
        if self._dry_run:
            return "DRY_RUN"
        return "LIVE"

    def _is_read_only_mode(self) -> bool:
        """Return True when the router must never place real orders."""
        return self._dry_run or self._is_validation_only_mode()

    def _requires_private_trading_access(self) -> bool:
        """Return True when authenticated private Polymarket access is required."""
        return self._is_validation_only_mode() or not self._dry_run

    @staticmethod
    def _parse_target_horizon_minutes(value: object) -> int:
        """Normalize model/market horizon metadata into a positive integer."""
        if isinstance(value, bool):
            return 0
        if isinstance(value, (int, float, str)):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return 0
            return parsed if parsed > 0 else 0
        return 0

    def _configured_model_horizons(self) -> list[int]:
        """Return all known target horizons across legacy and multi-model state."""
        horizons = sorted(
            horizon
            for horizon in self._models.keys()
            if isinstance(horizon, int) and horizon > 0
        )

        legacy_horizon = self._parse_target_horizon_minutes(
            getattr(self._model, "target_horizon_minutes", None)
        )
        if self._model is not None:
            legacy_horizon = legacy_horizon or DEFAULT_TARGET_HORIZON_MINUTES
            if legacy_horizon not in horizons:
                horizons.append(legacy_horizon)
                horizons.sort()

        return horizons

    def _active_pipelines(self) -> list[FeaturePipeline]:
        """Return currently configured pipelines, including legacy single-model state."""
        if self._pipelines:
            return list(self._pipelines.values())
        if self._pipeline is not None:
            return [self._pipeline]
        return []

    def _prediction_count(self) -> int:
        """Return total predictions emitted across loaded models."""
        if self._models:
            return sum(
                int(getattr(model, "prediction_count", 0))
                for model in self._models.values()
            )
        return int(getattr(self._model, "prediction_count", 0)) if self._model else 0

    def _resolve_inference_stack(
        self,
        target_horizon: int,
    ) -> tuple[Optional[object], Optional[object]]:
        """Resolve the best available model/pipeline pair for a target horizon."""
        if target_horizon in self._models and target_horizon in self._pipelines:
            return self._models[target_horizon], self._pipelines[target_horizon]

        if self._model is not None and self._pipeline is not None:
            legacy_horizon = (
                self._parse_target_horizon_minutes(
                    getattr(self._model, "target_horizon_minutes", None)
                )
                or DEFAULT_TARGET_HORIZON_MINUTES
            )
            if legacy_horizon == target_horizon or not self._models:
                return self._model, self._pipeline

        fallback_horizon = DEFAULT_TARGET_HORIZON_MINUTES
        if fallback_horizon in self._models and fallback_horizon in self._pipelines:
            return self._models[fallback_horizon], self._pipelines[fallback_horizon]

        if self._model is not None and self._pipeline is not None:
            return self._model, self._pipeline

        return None, None

    def _expected_market_interval_minutes(self) -> int:
        """Return the model's native target horizon for market compatibility checks."""
        interval = getattr(self._model, "target_horizon_minutes", None)
        parsed = self._parse_target_horizon_minutes(interval)
        if parsed > 0:
            return parsed
        configured_horizons = self._configured_model_horizons()
        if 5 in configured_horizons:
            return 5
        if configured_horizons:
            return configured_horizons[0]
        return DEFAULT_TARGET_HORIZON_MINUTES

    def _market_supports_live_strategy(self, market: MarketInfo) -> bool:
        """
        Fail closed when discovery finds a market with an incompatible horizon.

        Validation-only mode may still observe other market families, but live
        orders should not be routed into a market whose resolution interval does
        not match the primary loaded model's target horizon.
        """
        if self._is_read_only_mode():
            return True

        interval = getattr(market, "market_interval_minutes", None)
        expected_interval = self._expected_market_interval_minutes()
        if interval is None or interval == expected_interval:
            self._last_live_horizon_notice_key = None
            return True
        if getattr(TRADING, "allow_non_5m_live_markets", False):
            notice_key = (market.condition_id, "override", interval)
            if self._last_live_horizon_notice_key != notice_key:
                logger.warning(
                    "Live trading override enabled for incompatible market horizon | "
                    "slug=%s interval_minutes=%s expected=%s",
                    market.slug,
                    interval,
                    expected_interval,
                )
                self._last_live_horizon_notice_key = notice_key
            return True

        notice_key = (market.condition_id, "blocked", interval)
        if self._last_live_horizon_notice_key != notice_key:
            logger.error(
                "Refusing live trading on incompatible market horizon | slug=%s "
                "interval_minutes=%s expected=%s",
                market.slug,
                interval,
                expected_interval,
            )
            self._last_live_horizon_notice_key = notice_key
        return False

    @staticmethod
    def _market_poll_retry_interval_seconds() -> float:
        """Retry discovery faster when no executable market is currently active."""
        try:
            return max(float(getattr(POLYMARKET, "market_poll_retry_seconds", 5)), 1.0)
        except (TypeError, ValueError):
            return 5.0

    @staticmethod
    def _market_rejection_backoff_seconds() -> float:
        """Cooldown applied after rejecting a market as untradeable."""
        try:
            return max(
                float(getattr(POLYMARKET, "market_rejection_backoff_seconds", 15)),
                1.0,
            )
        except (TypeError, ValueError):
            return 15.0

    @staticmethod
    def _market_pathological_backoff_seconds() -> float:
        """Longer cooldown for dead books with clearly pathological quotes."""
        try:
            return max(
                float(
                    getattr(
                        POLYMARKET,
                        "market_pathological_backoff_seconds",
                        60,
                    )
                ),
                1.0,
            )
        except (TypeError, ValueError):
            return 60.0

    @staticmethod
    def _diagnostics_pathological(diagnostics: Optional[dict]) -> bool:
        """Detect whether discovery diagnostics indicate a dead/pathological book."""
        if not isinstance(diagnostics, dict):
            return False
        return bool(diagnostics.get("pathological"))

    @staticmethod
    def _market_rejection_key(market: MarketInfo) -> Optional[str]:
        """Return a stable identifier for per-market cooldowns."""
        return market.condition_id or market.slug or None

    @staticmethod
    def _market_family_key(market: MarketInfo) -> Optional[str]:
        """Group related markets so dead families can be skipped as a unit."""
        slug = (market.slug or "").lower()
        question = (market.question or "").lower()
        if slug.startswith("btc-updown-5m-"):
            return "btc-updown-5m"

        interval = TradingEngine._parse_target_horizon_minutes(
            getattr(market, "market_interval_minutes", None)
        )
        if interval == 60 or "hourly" in slug or "hourly" in question:
            return "btc-hourly-series"
        if interval == 5:
            return "btc-5m-listed"
        return None

    def _prune_market_family_rejection_backoffs(
        self,
        now_ts: Optional[float] = None,
    ) -> None:
        """Drop expired family cooldown entries."""
        now_ts = time.time() if now_ts is None else now_ts
        expired_keys = [
            key
            for key, backoff_until in self._market_family_rejection_backoff_until_by_key.items()
            if backoff_until <= now_ts
        ]
        for key in expired_keys:
            self._market_family_rejection_backoff_until_by_key.pop(key, None)

    def _market_family_rejection_remaining_seconds(
        self,
        family_key: Optional[str],
        now_ts: Optional[float] = None,
    ) -> float:
        """Return remaining cooldown for a rejected market family, if any."""
        if not family_key:
            return 0.0

        now_ts = time.time() if now_ts is None else now_ts
        self._prune_market_family_rejection_backoffs(now_ts)
        backoff_until = self._market_family_rejection_backoff_until_by_key.get(family_key)
        if backoff_until is None:
            return 0.0
        return max(backoff_until - now_ts, 0.0)

    def _market_family_rejection_sleep_seconds(
        self,
        family_key: Optional[str],
        markets: list[MarketInfo],
        now_ts: Optional[float] = None,
    ) -> float:
        """Sleep until family cooldown expires or the nearest market window rolls."""
        now_ts = time.time() if now_ts is None else now_ts
        remaining_backoff = self._market_family_rejection_remaining_seconds(
            family_key,
            now_ts,
        )
        if remaining_backoff <= 0:
            return self._market_poll_retry_interval_seconds()

        sleep_seconds = remaining_backoff
        end_times = [
            GammaAPIClient._parse_iso_timestamp(market.end_date)
            for market in markets
        ]
        future_end_times = [
            end_ts
            for end_ts in end_times
            if end_ts is not None and end_ts > now_ts
        ]
        if future_end_times:
            sleep_seconds = min(
                sleep_seconds,
                max(min(future_end_times) - now_ts, 1.0),
            )

        return max(sleep_seconds, 1.0)

    def _prune_market_rejection_backoffs(self, now_ts: Optional[float] = None) -> None:
        """Drop expired market cooldown entries."""
        now_ts = time.time() if now_ts is None else now_ts
        expired_keys = [
            key
            for key, backoff_until in self._market_rejection_backoff_until_by_key.items()
            if backoff_until <= now_ts
        ]
        for key in expired_keys:
            self._market_rejection_backoff_until_by_key.pop(key, None)

    def _market_rejection_remaining_seconds(
        self,
        market: MarketInfo,
        now_ts: Optional[float] = None,
    ) -> float:
        """Return remaining cooldown for a rejected market, if any."""
        now_ts = time.time() if now_ts is None else now_ts
        self._prune_market_rejection_backoffs(now_ts)

        key = self._market_rejection_key(market)
        if not key:
            return 0.0

        backoff_until = self._market_rejection_backoff_until_by_key.get(key)
        if backoff_until is None:
            return 0.0
        return max(backoff_until - now_ts, 0.0)

    def _market_rejection_sleep_seconds(
        self,
        market: MarketInfo,
        now_ts: Optional[float] = None,
    ) -> float:
        """
        Sleep until either cooldown expires or the market window rolls over.
        """
        now_ts = time.time() if now_ts is None else now_ts
        remaining_backoff = self._market_rejection_remaining_seconds(market, now_ts)
        if remaining_backoff <= 0:
            return self._market_poll_retry_interval_seconds()

        sleep_seconds = remaining_backoff
        market_end_ts = GammaAPIClient._parse_iso_timestamp(market.end_date)
        if market_end_ts is not None and market_end_ts > now_ts:
            sleep_seconds = min(sleep_seconds, max(market_end_ts - now_ts, 1.0))

        return max(sleep_seconds, 1.0)

    def _record_market_rejection(
        self,
        market: MarketInfo,
        diagnostics: Optional[dict],
        source: str,
        now_ts: Optional[float] = None,
    ) -> float:
        """Register a temporary cooldown for a market rejected as untradeable."""
        now_ts = time.time() if now_ts is None else now_ts
        key = self._market_rejection_key(market)
        is_pathological = self._diagnostics_pathological(diagnostics)
        cooldown_seconds = (
            self._market_pathological_backoff_seconds()
            if is_pathological
            else self._market_rejection_backoff_seconds()
        )
        if key:
            self._market_rejection_backoff_until_by_key[key] = now_ts + cooldown_seconds

        logger.info(
            "%s rejected market as untradeable | slug=%s rejection_class=%s backoff_seconds=%.1f diagnostics=%s",
            source,
            market.slug,
            "pathological" if is_pathological else "standard",
            cooldown_seconds,
            diagnostics,
        )
        return self._market_rejection_sleep_seconds(market, now_ts)

    def _record_market_family_rejection(
        self,
        family_key: Optional[str],
        markets: list[MarketInfo],
        source: str,
        now_ts: Optional[float] = None,
    ) -> float:
        """Register a temporary cooldown for an entire pathological market family."""
        if not family_key:
            return self._market_poll_retry_interval_seconds()

        now_ts = time.time() if now_ts is None else now_ts
        cooldown_seconds = self._market_pathological_backoff_seconds()
        self._market_family_rejection_backoff_until_by_key[family_key] = (
            now_ts + cooldown_seconds
        )
        logger.info(
            "%s cooled down pathological family | family=%s markets=%d backoff_seconds=%.1f",
            source,
            family_key,
            len(markets),
            cooldown_seconds,
        )
        return self._market_family_rejection_sleep_seconds(
            family_key,
            markets,
            now_ts,
        )

    def _clear_market_rejection(self, market: Optional[MarketInfo]) -> None:
        """Remove cooldown state once a market becomes usable again."""
        if market is None:
            return
        key = self._market_rejection_key(market)
        if key:
            self._market_rejection_backoff_until_by_key.pop(key, None)

    def _clear_market_family_rejection(self, market: Optional[MarketInfo]) -> None:
        """Remove family cooldown state once a market from that family is usable again."""
        if market is None:
            return
        family_key = self._market_family_key(market)
        if family_key:
            self._market_family_rejection_backoff_until_by_key.pop(family_key, None)

    @staticmethod
    def _discovery_rejection_reasons(diagnostics: Optional[dict]) -> list[str]:
        """Extract stable rejection reasons from per-side discovery diagnostics."""
        if not isinstance(diagnostics, dict):
            return ["unknown"]

        labels: set[str] = set()
        for side_key in ("yes", "no"):
            side_diagnostics = diagnostics.get(side_key)
            if not isinstance(side_diagnostics, dict):
                continue
            pathology_reason = side_diagnostics.get("pathology_reason")
            reason = side_diagnostics.get("reason")
            if pathology_reason:
                labels.add(str(pathology_reason))
            elif reason:
                labels.add(str(reason))

        if not labels:
            if diagnostics.get("pathological"):
                labels.add("pathological")
            elif diagnostics.get("executable") is False:
                labels.add("untradeable")

        return sorted(labels) or ["unknown"]

    @staticmethod
    def _format_discovery_reason_counts(reason_counts: Counter[str]) -> str:
        """Render aggregated discovery rejection reasons into a compact log field."""
        if not reason_counts:
            return "none"
        return ",".join(
            f"{reason}:{count}"
            for reason, count in reason_counts.most_common()
        )

    def _log_discovery_cycle_summary(
        self,
        *,
        candidates_total: int,
        families_total: int,
        markets_assessed: int,
        markets_rejected: int,
        family_backoff_skips: int,
        market_backoff_skips: int,
        pathological_families: int,
        selected_market_slug: Optional[str],
        sleep_interval: float,
        reason_counts: Counter[str],
    ) -> None:
        """Emit a single periodic summary line for each discovery cycle."""
        logger.info(
            "Discovery cycle summary | candidates=%d families=%d assessed=%d rejected=%d family_backoff_skips=%d market_backoff_skips=%d pathological_families=%d selected=%s next_poll=%.1fs reasons=%s",
            candidates_total,
            families_total,
            markets_assessed,
            markets_rejected,
            family_backoff_skips,
            market_backoff_skips,
            pathological_families,
            selected_market_slug or "none",
            sleep_interval,
            self._format_discovery_reason_counts(reason_counts),
        )

    def _assess_market_executability(
        self,
        market: MarketInfo,
    ) -> tuple[bool, Optional[dict]]:
        """
        Evaluate whether a discovered market should enter the active loop.

        Fail open if the router cannot provide diagnostics so the bot does not
        accidentally deadlock because of an observability bug.
        """
        if not self._router:
            return True, None

        assess_market = getattr(self._router, "assess_market_executability", None)
        if not callable(assess_market):
            return True, None

        try:
            diagnostics = assess_market(market)
        except Exception as exc:
            logger.warning(
                "Failed to assess market executability; keeping discovery permissive "
                "| slug=%s error=%s",
                market.slug,
                exc,
            )
            return True, None

        return bool(diagnostics.get("executable")), diagnostics

    def _get_market_discovery_candidates(self) -> list[MarketInfo]:
        """
        Return ordered discovery candidates from Gamma, falling back to the
        legacy single-market getter when the richer API is unavailable.
        """
        get_candidates = getattr(
            self._gamma,
            "get_active_btc_5m_market_candidates",
            None,
        )
        if callable(get_candidates):
            candidates = list(get_candidates(force_refresh=True) or [])
            if candidates:
                return candidates

        get_market = getattr(self._gamma, "get_active_btc_5m_market", None)
        if not callable(get_market):
            return []

        market = get_market(force_refresh=True)
        return [market] if market else []

    def _group_market_discovery_candidates(
        self,
        candidates: list[MarketInfo],
    ) -> list[tuple[str, list[MarketInfo]]]:
        """Preserve candidate order while grouping related markets into families."""
        grouped: dict[str, list[MarketInfo]] = {}
        for market in candidates:
            family_key = (
                self._market_family_key(market)
                or self._market_rejection_key(market)
                or f"market:{len(grouped)}"
            )
            grouped.setdefault(family_key, []).append(market)
        return list(grouped.items())

    def _rank_executable_market_candidate(
        self,
        market: MarketInfo,
        diagnostics: Optional[dict],
    ) -> tuple[float, float, float, float, str]:
        """
        Rank executable discovery candidates by practical live tradability.

        Preference order:
        1. Native-horizon markets first (5m over fallback hourly when both work).
        2. Tightest executable spread.
        3. Price closer to the middle band.
        4. Lower executable ask to preserve edge headroom.
        """
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        expected_interval = self._expected_market_interval_minutes()
        market_interval = self._parse_target_horizon_minutes(
            getattr(market, "market_interval_minutes", None)
        )
        native_horizon_penalty = (
            0.0 if market_interval in {0, expected_interval} else 1.0
        )
        best_spread = diagnostics.get("best_executable_spread")
        if best_spread is None:
            best_spread = float("inf")
        best_mid_price = diagnostics.get("best_executable_mid_price")
        center_distance = (
            abs(float(best_mid_price) - 0.5)
            if best_mid_price is not None
            else 1.0
        )
        best_price = diagnostics.get("best_executable_price")
        if best_price is None:
            best_price = float("inf")
        return (
            native_horizon_penalty,
            float(best_spread),
            center_distance,
            float(best_price),
            market.slug or market.condition_id or "",
        )

    def _refresh_active_market_if_needed(self) -> bool:
        """
        Refresh the active market when it is expired or close to expiry.

        Returns:
            True when a tradeable market is available after refresh, False otherwise.
        """
        if self._active_market is None:
            return False

        def reset_near_expiry_refresh() -> None:
            self._next_near_expiry_refresh_at = 0.0
            self._near_expiry_refresh_condition_id = None

        now_ts = time.time()
        market_end_ts = GammaAPIClient._parse_iso_timestamp(self._active_market.end_date)
        if market_end_ts is None:
            reset_near_expiry_refresh()
            return True

        expires_in = market_end_ts - now_ts
        should_refresh = expires_in <= 60
        if not should_refresh:
            reset_near_expiry_refresh()
            return True

        current_market = self._active_market
        current_condition_id = current_market.condition_id
        if (
            self._near_expiry_refresh_condition_id == current_condition_id
            and now_ts < self._next_near_expiry_refresh_at
        ):
            return expires_in > 0

        if expires_in <= 0:
            logger.warning(
                "Active market expired before inference | slug=%s end=%s",
                self._active_market.slug,
                self._active_market.end_date,
            )

        self._near_expiry_refresh_condition_id = current_condition_id
        self._next_near_expiry_refresh_at = (
            now_ts + self._near_expiry_refresh_cooldown_seconds
        )

        refreshed = self._gamma.get_active_btc_5m_market(force_refresh=True)
        if refreshed is None:
            return expires_in > 0

        refreshed_end_ts = GammaAPIClient._parse_iso_timestamp(refreshed.end_date)
        if refreshed_end_ts is not None and refreshed_end_ts <= now_ts:
            logger.warning(
                "Refreshed market is already expired | slug=%s end=%s",
                refreshed.slug,
                refreshed.end_date,
            )
            return expires_in > 0

        market_is_executable, diagnostics = self._assess_market_executability(
            refreshed
        )
        if not market_is_executable:
            self._record_market_rejection(
                refreshed,
                diagnostics,
                source="Near-expiry refresh",
                now_ts=now_ts,
            )
            self._set_active_market(None)
            reset_near_expiry_refresh()
            return False

        if current_condition_id != refreshed.condition_id:
            logger.info(
                "Refreshed near-expiry market | old=%s new=%s",
                current_market.slug,
                refreshed.slug,
            )
            reset_near_expiry_refresh()

        self._clear_market_rejection(refreshed)
        self._set_active_market(refreshed)
        if current_condition_id != refreshed.condition_id:
            asyncio.create_task(
                self._pm_ws.subscribe([refreshed.yes_token_id, refreshed.no_token_id])
            )
        return True

    def _run_private_connectivity_checks(
        self,
        require_spendable_collateral: bool = False,
    ) -> bool:
        """Verify authenticated account endpoints before starting sensitive modes."""
        if not self._pm_client:
            return False

        collateral = self._pm_client.get_collateral_balance_allowance()
        if collateral is None:
            logger.error(
                "Authenticated connectivity check failed: unable to read collateral "
                "balance/allowance"
            )
            return False

        open_orders = self._pm_client.get_open_orders() or []
        if collateral.available_to_trade <= 0:
            message = (
                "Authenticated Polymarket connectivity verified, but no spendable "
                "collateral is currently available"
            )
            if require_spendable_collateral:
                logger.error("%s", message)
                return False
            logger.warning("%s", message)

        logger.info(
            "Authenticated Polymarket connectivity verified | available_collateral=%.4f "
            "open_orders=%d",
            collateral.available_to_trade,
            len(open_orders),
        )
        return True

    def _initialize_clients(self) -> bool:
        """Initialize Polymarket client and dependent components."""
        try:
            read_only_mode = self._is_read_only_mode()
            requires_private_access = self._requires_private_trading_access()

            logger.info(
                "Runtime mode resolved | dry_run=%s validation_only=%s "
                "read_only=%s private_access=%s",
                self._dry_run,
                self._is_validation_only_mode(),
                read_only_mode,
                requires_private_access,
            )

            if not read_only_mode and not getattr(TRADING, "live_trading_enabled", False):
                logger.error(
                    "LIVE_TRADING_ENABLED is not set; refusing to start in live mode"
                )
                return False

            # Check required credentials
            if not getattr(POLYMARKET, "private_key", "") and requires_private_access:
                logger.error(
                    "POLYGON_PRIVATE_KEY not set in environment, but authenticated "
                    "Polymarket access is required"
                )
                return False

            if not getattr(POLYMARKET, "private_key", "") and self._dry_run:
                logger.warning(
                    "Starting in dry-run mode without a private key; "
                    "public market data will work, but live trading calls are disabled"
                )

            self._pm_client = PolymarketClient()

            if requires_private_access and not self._pm_client.has_trading_access:
                logger.error(
                    "Authenticated startup requires verified trading access "
                    "(private key plus CLOB credentials); refusing to start"
                )
                return False

            if requires_private_access and not self._run_private_connectivity_checks(
                require_spendable_collateral=not read_only_mode
            ):
                return False

            self._router = OrderRouter(self._pm_client, dry_run=read_only_mode)
            self._risk = RiskManager(
                self._state,
                self._pm_client,
                read_only_mode=read_only_mode,
            )
            self._position_manager = PositionManager(
                self._pm_client,
                read_only_mode=read_only_mode,
            )
            self._live_test_gate = self._build_live_test_gate()

            logger.info("All clients initialized successfully")
            return True
        except Exception as e:
            logger.error("Client initialization failed: %s", e)
            return False

    def _build_live_test_gate(self) -> Optional[LiveTestGate]:
        """
        Build the mandatory live shadow-testing gate when live trading is enabled.
        """
        if self._is_read_only_mode():
            return None

        if not getattr(TRADING, "require_live_test_before_live_orders", True):
            logger.warning(
                "Live qualification gate is disabled; live orders can start immediately"
            )
            return None

        gate = LiveTestGate(
            qualification_window_seconds=getattr(
                TRADING, "live_test_window_seconds", 600
            ),
            min_completed_markets=getattr(
                TRADING, "live_test_min_completed_markets", 2
            ),
            min_win_rate=getattr(TRADING, "live_test_min_win_rate", 0.50),
            min_profit=getattr(TRADING, "live_test_min_profit", 0.01),
            max_cumulative_loss=getattr(
                TRADING, "live_test_max_cumulative_loss", 0.0
            ),
            target_market_interval_minutes=self._expected_market_interval_minutes(),
        )
        logger.warning(
            "Live orders blocked pending shadow qualification | window=%ss "
            "min_markets=%d min_win_rate=%.2f min_profit=%.4f max_loss=%.4f",
            gate.qualification_window_seconds,
            gate.min_completed_markets,
            gate.min_win_rate,
            gate.min_profit,
            gate.max_cumulative_loss,
        )
        return gate

    @staticmethod
    def _result_represents_fill(result) -> bool:
        """
        Return True when an execution result should be treated like an entered position.

        Dry-run diagnostics can return success=True even when the order would be
        blocked live by venue minimums or collateral constraints. Those should
        not become managed positions or feed exit/P&L logic.
        """
        if not result or not getattr(result, "success", False):
            return False

        raw_response = getattr(result, "raw_response", None)
        if isinstance(raw_response, dict):
            if raw_response.get("live_blocked"):
                return False
            if raw_response.get("dry_run"):
                return bool(raw_response.get("simulated_fill"))

        return True

    @staticmethod
    def _result_requires_private_state_refresh(result) -> bool:
        """
        Refresh private account state only after paths that may have changed it.

        Local duplicate suppression and pre-submit guardrails do not touch
        Polymarket account state, so invalidating the cache there just creates
        avoidable private-endpoint churn in the live loop.
        """
        if result is None:
            return False

        raw_response = getattr(result, "raw_response", None)
        if isinstance(raw_response, dict):
            if raw_response.get("dry_run"):
                return False
            if raw_response.get("reason") in {
                "order_sizing_blocked",
                "insufficient_balance_or_allowance",
            }:
                return False

        # Successful live submissions and non-local failures should refresh
        # conservatively because account/order state may have changed.
        return True

    def _load_model(self) -> bool:
        """Load all available trained LightGBM models."""
        for horizon in [5, 60]:
            from config.settings import PATHS
            import os
            model_path = os.path.join(PATHS.models_dir, f"lgbm_btc_{horizon}m.txt")
            if os.path.exists(model_path):
                model = ModelInference(model_path=model_path)
                if model.load():
                    self._models[horizon] = model
                    model_features = model.metadata.get("feature_columns")
                    if model_features:
                        self._pipelines[horizon] = FeaturePipeline(
                            self._state, model_feature_columns=model_features
                        )
                        logger.info(
                            "Pipeline configured for %d model features (horizon %dm)", 
                            len(model_features), horizon
                        )

        if not self._models:
            logger.error("No models loaded! Train models before running.")
            return False

        primary_horizon = 5 if 5 in self._models else min(self._models)
        self._model = self._models[primary_horizon]
        self._pipeline = self._pipelines.get(primary_horizon)
            
        return True

    def _build_runtime_summary(self) -> dict:
        """Return a compact end-of-run summary for audit manifests."""
        pipelines = self._active_pipelines()
        summary = {
            "mode": self._mode_label(),
            "running": self._running,
            "state_size": self._state.size,
            "minute_bars": pipelines[0].minute_bar_count if pipelines else 0,
            "prediction_count": self._prediction_count(),
            "model_target_horizon_minutes": self._configured_model_horizons(),
            "active_market": None,
        }

        if self._active_market is not None:
            summary["active_market"] = {
                "slug": self._active_market.slug,
                "condition_id": self._active_market.condition_id,
                "end_date": self._active_market.end_date,
                "market_interval_minutes": self._active_market.market_interval_minutes,
            }

        if self._router:
            summary["router"] = {
                "orders_placed": self._router.orders_placed,
                "orders_rejected": self._router.orders_rejected,
                "orders_simulated": self._router.orders_simulated,
                "duplicate_signals_suppressed": (
                    self._router.duplicate_signals_suppressed
                ),
            }

        if self._risk:
            summary["risk"] = self._risk.get_status()

        if self._position_manager:
            summary["position_manager"] = self._position_manager.get_status()

        if self._live_test_gate:
            summary["live_test"] = self._live_test_gate.get_status()

        return summary

    def _finalize_run_manifest(self, status: str, error: Optional[str] = None):
        """Best-effort manifest finalization for auditability."""
        if not self._run_manifest_manager:
            return

        try:
            self._run_manifest_manager.finalize(
                status=status,
                runtime_summary=self._build_runtime_summary(),
                error=error,
            )
        except Exception as e:
            logger.error("Failed to finalize run manifest: %s", e)

    def _seed_pipeline_history(self):
        """
        Backfill recent closed Binance minute bars so inference can start promptly.
        """
        pipelines = self._active_pipelines()
        target_bars = max(
            (p.required_complete_bars for p in pipelines),
            default=1,
        ) + 18
        try:
            bars = self._binance_rest.fetch_recent_1m_klines(limit=target_bars)
        except Exception as e:
            logger.warning("Historical Binance warmup fetch failed: %s", e)
            return

        if bars.empty:
            logger.warning("Historical Binance warmup returned no closed minute bars")
            return

        for p in pipelines:
            p.seed_historical_bars(bars)
        
        logger.info(
            "Seeded pipeline minute-bar history | bars=%d range=%s -> %s",
            len(bars),
            bars["open_time"].iloc[0],
            bars["open_time"].iloc[-1],
        )

    async def _binance_consumer_task(self):
        """Task 1: Run the Binance WebSocket consumer."""
        logger.info("Starting Binance WebSocket consumer...")
        try:
            await self._binance_ws.start()
        except asyncio.CancelledError:
            await self._binance_ws.stop()
            logger.info("Binance consumer stopped")

    async def _state_ingestion_task(self):
        """Task 2: Move data from the WS queue into the RollingState."""
        logger.info("Starting state ingestion loop...")
        queue = self._binance_ws.queue

        try:
            while self._running:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    await self._state.push_event(event)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info("State ingestion stopped")

    async def _market_discovery_task(self):
        """Task 3: Poll Gamma API for the active BTC market."""
        logger.info("Starting market discovery loop...")
        poll_interval = POLYMARKET.market_poll_interval_seconds
        retry_interval = self._market_poll_retry_interval_seconds()

        try:
            while self._running:
                sleep_interval = poll_interval
                try:
                    candidates = self._get_market_discovery_candidates()
                    reason_counts: Counter[str] = Counter()
                    markets_assessed = 0
                    markets_rejected = 0
                    family_backoff_skips = 0
                    market_backoff_skips = 0
                    pathological_families = 0
                    selected_market_slug: Optional[str] = None
                    families_total = 0
                    executable_candidates: list[
                        tuple[
                            tuple[float, float, float, float, str],
                            int,
                            int,
                            MarketInfo,
                            Optional[dict],
                        ]
                    ] = []
                    if candidates:
                        now_ts = time.time()
                        selected_market: Optional[MarketInfo] = None
                        best_retry_sleep: Optional[float] = None
                        grouped_candidates = self._group_market_discovery_candidates(
                            candidates
                        )
                        families_total = len(grouped_candidates)

                        candidate_rank = 0
                        for family_index, (family_key, family_markets) in enumerate(
                            grouped_candidates,
                            start=1,
                        ):
                            remaining_family_backoff = (
                                self._market_family_rejection_remaining_seconds(
                                    family_key,
                                    now_ts,
                                )
                            )
                            if remaining_family_backoff > 0:
                                family_backoff_skips += 1
                                reason_counts["family_backoff"] += 1
                                logger.info(
                                    "Discovery skipped cooled-down family | family=%s rank=%d/%d remaining_backoff=%.1f candidates=%d",
                                    family_key,
                                    family_index,
                                    len(grouped_candidates),
                                    remaining_family_backoff,
                                    len(family_markets),
                                )
                                self._set_active_market(None)
                                family_sleep = self._market_family_rejection_sleep_seconds(
                                    family_key,
                                    family_markets,
                                    now_ts,
                                )
                                best_retry_sleep = (
                                    family_sleep
                                    if best_retry_sleep is None
                                    else min(best_retry_sleep, family_sleep)
                                )
                                candidate_rank += len(family_markets)
                                continue

                            family_all_pathological = True
                            family_assessed = False

                            for market in family_markets:
                                candidate_rank += 1
                                remaining_rejection_backoff = (
                                    self._market_rejection_remaining_seconds(
                                        market,
                                        now_ts,
                                    )
                                )
                                if remaining_rejection_backoff > 0:
                                    market_backoff_skips += 1
                                    reason_counts["market_backoff"] += 1
                                    logger.info(
                                        "Discovery skipped cooled-down market | slug=%s rank=%d/%d remaining_backoff=%.1f",
                                        market.slug,
                                        candidate_rank,
                                        len(candidates),
                                        remaining_rejection_backoff,
                                    )
                                    self._set_active_market(None)
                                    candidate_sleep = self._market_rejection_sleep_seconds(
                                        market,
                                        now_ts,
                                    )
                                    best_retry_sleep = (
                                        candidate_sleep
                                        if best_retry_sleep is None
                                        else min(best_retry_sleep, candidate_sleep)
                                    )
                                    continue

                                if not self._market_supports_live_strategy(market):
                                    markets_rejected += 1
                                    reason_counts["unsupported_live_horizon"] += 1
                                    self._set_active_market(None)
                                    best_retry_sleep = (
                                        retry_interval
                                        if best_retry_sleep is None
                                        else min(best_retry_sleep, retry_interval)
                                    )
                                    continue

                                family_assessed = True
                                markets_assessed += 1
                                market_is_executable, diagnostics = (
                                    self._assess_market_executability(market)
                                )
                                if not self._diagnostics_pathological(diagnostics):
                                    family_all_pathological = False

                                if not market_is_executable:
                                    markets_rejected += 1
                                    for reason in self._discovery_rejection_reasons(
                                        diagnostics
                                    ):
                                        reason_counts[reason] += 1
                                    candidate_sleep = self._record_market_rejection(
                                        market,
                                        diagnostics,
                                        source="Discovery",
                                        now_ts=now_ts,
                                    )
                                    self._set_active_market(None)
                                    best_retry_sleep = (
                                        candidate_sleep
                                        if best_retry_sleep is None
                                        else min(best_retry_sleep, candidate_sleep)
                                    )
                                    continue

                                executable_candidates.append(
                                    (
                                        self._rank_executable_market_candidate(
                                            market,
                                            diagnostics,
                                        ),
                                        candidate_rank,
                                        family_index,
                                        market,
                                        diagnostics,
                                    )
                                )
                                family_all_pathological = False

                            if family_assessed and family_all_pathological:
                                pathological_families += 1
                                reason_counts["family_all_pathological"] += 1
                                family_sleep = self._record_market_family_rejection(
                                    family_key,
                                    family_markets,
                                    source="Discovery",
                                    now_ts=now_ts,
                                )
                                best_retry_sleep = (
                                    family_sleep
                                    if best_retry_sleep is None
                                    else min(best_retry_sleep, family_sleep)
                                )

                        if executable_candidates:
                            executable_candidates.sort(
                                key=lambda item: (
                                    item[0],
                                    item[1],
                                    item[2],
                                )
                            )
                            selected_rank = executable_candidates[0][0]
                            selected_candidate_rank = executable_candidates[0][1]
                            selected_family_rank = executable_candidates[0][2]
                            selected_market = executable_candidates[0][3]
                            selected_market_slug = selected_market.slug
                            self._clear_market_rejection(selected_market)
                            self._clear_market_family_rejection(selected_market)
                            if selected_candidate_rank > 1:
                                logger.info(
                                    "Discovery selected ranked executable market | slug=%s family=%s rank=%d/%d family_rank=%d/%d spread=%.4f center_distance=%.4f best_price=%.4f native_horizon_penalty=%.1f",
                                    selected_market.slug,
                                    self._market_family_key(selected_market)
                                    or "unknown",
                                    selected_candidate_rank,
                                    len(candidates),
                                    selected_family_rank,
                                    len(grouped_candidates),
                                    selected_rank[1],
                                    selected_rank[2],
                                    selected_rank[3],
                                    selected_rank[0],
                                )

                        if selected_market is not None:
                            if (
                                self._active_market is None
                                or selected_market.condition_id
                                != self._active_market.condition_id
                            ):
                                logger.info(
                                    "New active market | slug=%s yes=%s",
                                    selected_market.slug,
                                    selected_market.yes_token_id[:16] + "...",
                                )
                                asyncio.create_task(
                                    self._pm_ws.subscribe(
                                        [
                                            selected_market.yes_token_id,
                                            selected_market.no_token_id,
                                        ]
                                    )
                                )
                            self._set_active_market(selected_market)
                        else:
                            self._set_active_market(None)
                            logger.info(
                                "No executable BTC market candidate found; inference remains paused"
                            )
                            if best_retry_sleep is not None:
                                sleep_interval = best_retry_sleep
                            else:
                                sleep_interval = retry_interval
                    else:
                        logger.warning("No active BTC market found")
                        reason_counts["no_candidates"] += 1
                        self._set_active_market(None)
                        sleep_interval = retry_interval

                    self._log_discovery_cycle_summary(
                        candidates_total=len(candidates),
                        families_total=families_total,
                        markets_assessed=markets_assessed,
                        markets_rejected=markets_rejected,
                        family_backoff_skips=family_backoff_skips,
                        market_backoff_skips=market_backoff_skips,
                        pathological_families=pathological_families,
                        selected_market_slug=selected_market_slug,
                        sleep_interval=sleep_interval,
                        reason_counts=reason_counts,
                    )

                except Exception as e:
                    logger.error("Market discovery error: %s", e)
                    sleep_interval = retry_interval

                await asyncio.sleep(sleep_interval)
        except asyncio.CancelledError:
            logger.info("Market discovery stopped")

    async def _inference_trading_task(self):
        """
        Task 4: Main inference and trading loop.
        On each tick (when new data arrives), compute features, run inference,
        and route orders if an edge is found.
        """
        logger.info("Starting inference & trading loop...")
        inference_interval = 1.0  # seconds between inference cycles

        # Wait until all feature pipelines have enough closed minute bars.
        while self._running:
            pipelines = self._active_pipelines()
            if pipelines and all(p.is_ready for p in pipelines):
                break
            if pipelines:
                p = pipelines[0]
                logger.debug(
                    "Warming up feature pipelines... minute_bars=%d trades=%d",
                    p.minute_bar_count,
                    self._state.size,
                )
            await asyncio.sleep(2)

        if not self._running:
            return

        logger.info(
            "Feature pipelines ready — beginning inference | minute_bars=%d",
            self._active_pipelines()[0].minute_bar_count
            if self._active_pipelines()
            else 0,
        )

        try:
            while self._running:
                if not self._market_ready_event.is_set():
                    if not self._inference_waiting_for_market_logged:
                        logger.info(
                            "Inference paused waiting for executable market candidate"
                        )
                        self._inference_waiting_for_market_logged = True
                    try:
                        await asyncio.wait_for(
                            self._market_ready_event.wait(),
                            timeout=inference_interval,
                        )
                    except asyncio.TimeoutError:
                        continue

                cycle_start = time.time()

                try:
                    await self._run_inference_cycle()
                except Exception as e:
                    logger.error("Inference cycle error: %s", e)

                # Sleep for the remainder of the interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0, inference_interval - elapsed)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.info("Inference loop stopped")

    async def _run_inference_cycle(self):
        """Single inference + trade evaluation cycle."""
        if self._position_manager:
            for exit_execution in self._position_manager.evaluate_positions(self._state):
                is_dry_run_exit = bool(
                    exit_execution.result.raw_response
                    and exit_execution.result.raw_response.get("dry_run")
                )
                if (
                    is_dry_run_exit
                    and self._risk
                    and not self._is_read_only_mode()
                    and abs(exit_execution.realized_pnl) > 1e-9
                ):
                    self._risk.update_pnl(exit_execution.realized_pnl)

        # Need an active market to trade
        if self._active_market is None:
            return

        if not self._refresh_active_market_if_needed():
            return

        if not self._market_supports_live_strategy(self._active_market):
            return

        should_poll_shadow_signals = False
        if self._live_test_gate:
            self._live_test_gate.settle_due_trades(self._state)
            should_poll_shadow_signals = (
                not self._live_test_gate.allows_live_trading
                and self._live_test_gate.accepts_new_signals
            )
            if (
                not self._live_test_gate.allows_live_trading
                and not self._live_test_gate.accepts_new_signals
            ):
                return

        if self._risk and not should_poll_shadow_signals:
            if not self._risk.run_all_checks(
                include_balance_check=not self._is_read_only_mode(),
                include_position_limit=not self._is_read_only_mode(),
            ):
                return

        # Route to appropriate model
        target_horizon = self._parse_target_horizon_minutes(
            getattr(self._active_market, "market_interval_minutes", None)
        )
        if target_horizon <= 0:
            target_horizon = self._expected_market_interval_minutes()

        model, pipeline = self._resolve_inference_stack(target_horizon)
        if model is None or pipeline is None:
            logger.error("No model available for fallback.")
            return

        # Compute features
        features = pipeline.compute()
        if features is None:
            return

        # Run inference
        raw_prob = model.predict(features)
        if raw_prob is None:
            return

        prob = self._probability_estimator.estimate_yes_probability(
            raw_prob,
            self._active_market,
            self._state,
        )

        if not self._router:
            return

        signal = self._router.get_signal(prob, self._active_market)

        # Improvement 2: EMA trend confirmation filter
        if signal is not None and not self._trend_filter.confirms_direction(
            signal.side, self._state
        ):
            logger.debug(
                "Signal blocked by trend filter | side=%s market=%s",
                signal.side,
                self._active_market.slug,
            )
            signal = None

        if self._live_test_gate and not self._live_test_gate.allows_live_trading:
            if signal is not None:
                self._live_test_gate.record_shadow_signal(self._active_market, signal)
            if not self._live_test_gate.allows_live_trading:
                return

        if signal is None:
            return

        result = self._router.execute_signal(signal, self._active_market)
        if self._risk and self._result_requires_private_state_refresh(result):
            invalidate_private_cache = getattr(
                self._risk, "invalidate_private_check_cache", None
            )
            if callable(invalidate_private_cache):
                invalidate_private_cache()
        if result and result.success:
            is_fill = self._result_represents_fill(result)
            if self._position_manager and is_fill:
                self._position_manager.record_entry(signal, self._active_market)
            is_dry_run = bool(result.raw_response and result.raw_response.get("dry_run"))
            logger.info(
                "%s | p̂=%.4f raw_p̂=%.4f market=%s",
                (
                    "Dry-run trade simulated"
                    if is_dry_run and is_fill
                    else "Dry-run signal blocked"
                    if is_dry_run
                    else "Trade executed"
                ),
                prob,
                raw_prob,
                self._active_market.slug,
            )

    async def _risk_monitor_task(self):
        """Task 5: Continuous risk monitoring."""
        logger.info("Starting risk monitor...")
        check_interval = 5.0  # Check every 5 seconds

        try:
            while self._running:
                if self._live_test_gate and not self._live_test_gate.allows_live_trading:
                    await asyncio.sleep(check_interval)
                    continue

                if self._risk:
                    triggered = self._risk.check_volatility()
                    if triggered:
                        logger.critical("KILL-SWITCH ACTIVE — orders cancelled")

                    # Log status periodically
                    if int(time.time()) % 60 == 0:
                        status = self._risk.get_status()
                        logger.info("Risk status: %s", status)

                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            logger.info("Risk monitor stopped")

    def _sync_realized_pnl(self, initialize_only: bool = False) -> float:
        """
        Pull realized P&L from Polymarket positions and apply only the delta.

        The Data API reports cumulative `realizedPnl` per position. We treat the
        first snapshot after startup as a session baseline so the bot's risk
        floor is not polluted by wallet history from older manual/bot trades.
        """
        if not self._pm_client or not self._risk:
            return 0.0

        current_positions = self._pm_client.get_current_positions(limit=50) or []
        closed_positions = self._pm_client.get_closed_positions(limit=50) or []
        if not current_positions and not closed_positions:
            return 0.0

        latest_realized_pnl: dict[str, tuple[float, int]] = {}
        for position in [*current_positions, *closed_positions]:
            if not isinstance(position, dict):
                continue

            position_key = self._position_tracking_key(position)
            if position_key is None:
                continue

            realized_pnl = self._coerce_float(position.get("realizedPnl"))
            timestamp = self._coerce_int(position.get("timestamp"))
            prior_snapshot = latest_realized_pnl.get(position_key)
            if prior_snapshot is None or timestamp >= prior_snapshot[1]:
                latest_realized_pnl[position_key] = (realized_pnl, timestamp)

        if initialize_only or not self._realized_pnl_baseline_initialized:
            self._tracked_realized_pnl_by_position = {
                position_key: realized_pnl
                for position_key, (realized_pnl, _) in latest_realized_pnl.items()
            }
            self._realized_pnl_baseline_initialized = True
            logger.info(
                "Initialized realized P&L session baseline | tracked_positions=%d",
                len(self._tracked_realized_pnl_by_position),
            )
            return 0.0

        pnl_delta = 0.0
        for position_key, (realized_pnl, _) in latest_realized_pnl.items():
            previous_realized = self._tracked_realized_pnl_by_position.get(position_key)
            self._tracked_realized_pnl_by_position[position_key] = realized_pnl

            if previous_realized is None:
                # Position appeared after our baseline snapshot — could be a
                # pre-existing wallet position that wasn't in the first API
                # page. Treat its current P&L as the new baseline instead of
                # counting the full amount as session delta.
                logger.debug(
                    "New position appeared post-baseline | key=%s realized=%.4f (baselined)",
                    position_key[:24],
                    realized_pnl,
                )
            else:
                pnl_delta += realized_pnl - previous_realized

        if abs(pnl_delta) <= 1e-9:
            return 0.0

        self._risk.update_pnl(pnl_delta)
        logger.info(
            "Realized P&L synchronized | delta=%.4f tracked_positions=%d",
            pnl_delta,
            len(self._tracked_realized_pnl_by_position),
        )
        return pnl_delta

    @staticmethod
    def _position_tracking_key(position: dict) -> Optional[str]:
        """Build a stable deduplication key for Data API position snapshots."""
        asset = str(position.get("asset") or "").strip()
        owner = str(
            position.get("proxyWallet")
            or position.get("user")
            or ""
        ).strip()
        if asset:
            return f"{owner}:{asset}" if owner else asset

        condition_id = str(position.get("conditionId") or "").strip()
        outcome = str(
            position.get("outcome")
            or position.get("outcomeIndex")
            or ""
        ).strip()
        if condition_id and outcome:
            return f"{owner}:{condition_id}:{outcome}" if owner else f"{condition_id}:{outcome}"
        return None

    @staticmethod
    def _coerce_float(value: object) -> float:
        """Parse numeric Data API values defensively."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_int(value: object) -> int:
        """Parse integer-ish Data API fields defensively."""
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    async def _pnl_tracking_task(self):
        """Task 6: Periodically track realized P&L from filled orders."""
        if self._is_read_only_mode():
            return

        logger.info("Starting P&L tracking loop...")
        poll_interval = 300  # 5 minutes

        try:
            while self._running:
                try:
                    self._sync_realized_pnl()
                except Exception as e:
                    logger.error("P&L tracking error: %s", e)
                
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            logger.info("P&L tracking stopped")

    async def _status_reporter_task(self):
        """Periodic status logging."""
        try:
            while self._running:
                await asyncio.sleep(300)  # Every 5 minutes

                pipelines = self._active_pipelines()
                status = {
                    "state_size": self._state.size,
                    "state_maxlen": self._state.maxlen,
                    "history_span_seconds": round(self._state.history_span_seconds, 1),
                    "minute_bars": pipelines[0].minute_bar_count if pipelines else 0,
                    "pipeline_ready": all(p.is_ready for p in pipelines),
                    "last_price": self._state.last_price,
                    "trades_received": self._state.trade_count,
                    "predictions": self._prediction_count(),
                    "dry_run": self._dry_run,
                    "validation_only_mode": self._is_validation_only_mode(),
                    "active_market": (
                        self._active_market.slug if self._active_market else "None"
                    ),
                }
                
                if self._router:
                    status["orders_placed"] = self._router.orders_placed
                    status["orders_rejected"] = self._router.orders_rejected
                    status["orders_simulated"] = self._router.orders_simulated
                    status["duplicate_signals_suppressed"] = (
                        self._router.duplicate_signals_suppressed
                    )
                    status["filter_rejections"] = self._router.filter_stats

                if self._live_test_gate:
                    status["live_test"] = self._live_test_gate.get_status()

                if self._position_manager:
                    status.update(self._position_manager.get_status())
                    
                if self._risk:
                    status.update(self._risk.get_status())

                logger.info("STATUS REPORT: %s", status)
        except asyncio.CancelledError:
            pass

    async def run(self):
        """
        Main entry point. Initializes all components and launches concurrent tasks.
        """
        setup_logging(
            level=os.getenv("LOG_LEVEL", "INFO"),
            json_output=os.getenv("LOG_JSON", "false").lower() == "true",
        )

        logger.info("=" * 60)
        mode_label = self._mode_label()
        logger.info(
            "Polymarket BTC Trading Bot — Starting (%s mode)",
            mode_label,
        )
        logger.info("=" * 60)

        try:
            self._run_manifest_manager = RunManifestManager()
            self._run_manifest_manager.start(
                mode_label=mode_label,
                config_snapshot=build_runtime_config_snapshot(
                    dry_run=self._dry_run,
                    validation_only=self._is_validation_only_mode(),
                ),
            )
            logger.info(
                "Run manifest initialized | run_id=%s path=%s",
                self._run_manifest_manager.run_id,
                self._run_manifest_manager.manifest_path,
            )
        except Exception as e:
            logger.error("Cannot initialize run manifest. Exiting. error=%s", e)
            return

        try:
            validate_runtime_configuration(
                dry_run=self._dry_run,
                validation_only=self._is_validation_only_mode(),
            )
        except RuntimeConfigurationError as e:
            logger.error("Runtime configuration invalid: %s", e)
            self._finalize_run_manifest("startup_failed", error=str(e))
            return

        # Initialize
        if not self._load_model():
            logger.error("Cannot start without a trained model. Exiting.")
            self._finalize_run_manifest("startup_failed", error="model_load_failed")
            return

        if not self._initialize_clients():
            logger.error("Cannot start without valid credentials. Exiting.")
            self._finalize_run_manifest(
                "startup_failed",
                error="client_initialization_failed",
            )
            return

        if not self._is_read_only_mode():
            self._sync_realized_pnl(initialize_only=True)

        self._seed_pipeline_history()

        self._running = True

        # Install signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        # Launch all concurrent tasks
        tasks = [
            asyncio.create_task(self._binance_consumer_task(), name="binance_ws"),
            asyncio.create_task(self._state_ingestion_task(), name="state_ingest"),
            asyncio.create_task(self._market_discovery_task(), name="market_disc"),
            asyncio.create_task(self._inference_trading_task(), name="inference"),
            asyncio.create_task(self._risk_monitor_task(), name="risk_monitor"),
            asyncio.create_task(self._status_reporter_task(), name="status"),
            asyncio.create_task(self._pnl_tracking_task(), name="pnl_tracker"),
            asyncio.create_task(self._pm_ws.start(), name="pm_ws"),
        ]

        logger.info("All %d tasks launched", len(tasks))
        try:
            self._run_manifest_manager.mark_running(
                runtime_summary=self._build_runtime_summary()
            )
        except Exception as e:
            logger.error("Failed to update run manifest to running: %s", e)

        task_failure_message: Optional[str] = None
        try:
            # Wait for shutdown signal or any task failure
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )

            # Check for exceptions
            for task in done:
                if task.exception():
                    task_failure_message = (
                        f"Task '{task.get_name()}' failed: {task.exception()}"
                    )
                    logger.error(
                        "Task '%s' failed: %s",
                        task.get_name(),
                        task.exception(),
                    )
        finally:
            await self.shutdown()

            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
            self._finalize_run_manifest(
                "task_failed" if task_failure_message else "shutdown_complete",
                error=task_failure_message,
            )
            logger.info("All tasks stopped. Engine shutdown complete.")

    async def shutdown(self):
        """Graceful shutdown sequence."""
        if not self._running:
            return

        logger.info("Initiating graceful shutdown...")
        self._running = False

        # Cancel all resting orders
        if self._pm_client and not self._is_read_only_mode():
            logger.info("Cancelling all resting orders...")
            self._pm_client.cancel_all_orders()
        elif self._pm_client:
            logger.info("Read-only mode active; skipping cancel_all_orders on shutdown")

        # Close connections
        await self._binance_ws.stop()
        await self._pm_ws.stop()
        self._binance_rest.close()
        self._gamma.close()

        logger.info("Shutdown complete")


def main():
    """Entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Polymarket BTC Trading Bot")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Run in simulation mode (no live trades)"
    )
    parser.add_argument(
        "--live", 
        action="store_true", 
        help="Force live trading mode"
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Use authenticated account checks and live market data without placing orders",
    )
    args = parser.parse_args()

    # Determine dry_run intent
    dry_run = None
    if args.dry_run:
        dry_run = True
    elif args.live:
        dry_run = False

    engine = TradingEngine(
        dry_run_override=dry_run,
        validation_only_override=True if args.validation_only else None,
    )
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
