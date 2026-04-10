"""
Gamma API client for discovering active Polymarket BTC markets.

The original v1 target was the short-dated BTC 5-minute contract. Polymarket's
live market surface currently appears to favor hourly recurring BTC contracts,
so discovery keeps the old matcher and falls back to the active hourly series.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from config.settings import POLYMARKET

logger = logging.getLogger(__name__)


@dataclass
class MarketInfo:
    """Information about an active BTC market."""
    condition_id: str
    question: str
    slug: str
    yes_token_id: str
    no_token_id: str
    end_date: str
    active: bool = True
    neg_risk: bool = False
    indicative_yes_price: Optional[float] = None
    indicative_no_price: Optional[float] = None
    min_order_size: Optional[float] = None
    market_interval_minutes: Optional[int] = None
    fetched_at: float = field(default_factory=time.time)

    def is_stale(self, max_age_seconds: int = 240) -> bool:
        """Check if this market info is too old and needs refresh."""
        return (time.time() - self.fetched_at) > max_age_seconds


class GammaAPIClient:
    """
    Client for the Polymarket Gamma API.
    Discovers active BTC markets and their token IDs.
    """

    def __init__(self, base_url: Optional[str] = None):
        self._base_url = base_url or POLYMARKET.gamma_api_base
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "polymarket-btc-bot/0.1",
        })
        self._cached_market: Optional[MarketInfo] = None
        self._event_page_backoff_until_by_slug: dict[str, float] = {}
        self._event_page_backoff_seconds = 300.0
        self._btc_updown_interval_seconds = 300

    def get_active_btc_5m_market(self, force_refresh: bool = False) -> Optional[MarketInfo]:
        """
        Find the currently active BTC resolution market.

        We prefer the original 5-minute contract when available. If Polymarket
        is currently surfacing the newer hourly recurring Bitcoin contract
        family instead, we fall back to that live series and choose the child
        market closest to a 50/50 binary threshold.

        Uses caching to avoid excessive API calls.
        
        Returns:
            MarketInfo if found, None otherwise.
        """
        # Return cache if still fresh
        if (
            not force_refresh
            and self._cached_market is not None
            and not self._cached_market.is_stale()
        ):
            return self._cached_market

        try:
            if POLYMARKET.event_slug:
                market = self._fetch_market_from_event_slug(POLYMARKET.event_slug)
            else:
                market = self._fetch_btc_5m_market()
            if market:
                self._cached_market = market
                logger.info(
                    "Configured BTC market found | slug=%s yes=%s no=%s",
                    market.slug,
                    market.yes_token_id[:12] + "...",
                    market.no_token_id[:12] + "...",
                )
            else:
                logger.warning("No configured BTC market found")
            return market

        except Exception as e:
            logger.error("Gamma API error: %s", e)
            # Return stale cache rather than None if available
            if self._cached_market is not None:
                logger.warning("Returning stale cached market data")
                return self._cached_market
            return None

    def get_active_btc_5m_market_candidates(
        self,
        force_refresh: bool = False,
        limit: int = 10,
    ) -> list[MarketInfo]:
        """
        Return ordered BTC market candidates for discovery-time selection.

        Candidates are sorted by nearest relevant window first so callers can
        iterate until they find the first executable market.
        """
        if limit <= 0:
            return []

        if (
            not force_refresh
            and self._cached_market is not None
            and not self._cached_market.is_stale()
        ):
            return [self._cached_market]

        try:
            if POLYMARKET.event_slug:
                candidates = self._fetch_markets_from_event_slug(
                    POLYMARKET.event_slug,
                    limit=limit,
                )
            else:
                candidates = self._fetch_btc_5m_market_candidates(limit=limit)

            if candidates:
                self._cached_market = candidates[0]
            return candidates[:limit]

        except Exception as e:
            logger.error("Gamma API candidate fetch error: %s", e)
            if self._cached_market is not None:
                logger.warning("Returning stale cached candidate market data")
                return [self._cached_market]
            return []

    def _fetch_market_from_event_slug(
        self,
        event_slug: str,
        market_interval_minutes: Optional[int] = None,
    ) -> Optional[MarketInfo]:
        """Fetch the best-ranked live market from a Polymarket event page."""
        markets = self._fetch_markets_from_event_slug(
            event_slug,
            market_interval_minutes=market_interval_minutes,
            limit=1,
        )
        return markets[0] if markets else None

    def _fetch_markets_from_event_slug(
        self,
        event_slug: str,
        market_interval_minutes: Optional[int] = None,
        limit: int = 1,
    ) -> list[MarketInfo]:
        """Fetch one or more ranked live markets from a Polymarket event page."""
        if limit <= 0:
            return []

        if self._is_event_page_fetch_backed_off(event_slug):
            return self._fetch_markets_from_event_api_fallback(
                event_slug,
                market_interval_minutes=market_interval_minutes,
                limit=limit,
            )

        url = f"{POLYMARKET.event_base}/{event_slug}"
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(
                "Event page fetch failed for slug=%s; trying API fallback: %s",
                event_slug,
                e,
            )
            self._back_off_event_page_fetch(event_slug)
            return self._fetch_markets_from_event_api_fallback(
                event_slug,
                market_interval_minutes=market_interval_minutes,
                limit=limit,
            )

        page_data = self._extract_next_data(resp.text)
        if page_data is None:
            logger.warning(
                "No __NEXT_DATA__ payload found for slug=%s; trying API fallback",
                event_slug,
            )
            self._back_off_event_page_fetch(event_slug)
            return self._fetch_markets_from_event_api_fallback(
                event_slug,
                market_interval_minutes=market_interval_minutes,
                limit=limit,
            )

        event_payload = self._extract_event_payload(page_data, event_slug)
        if event_payload is not None:
            event_markets = self._select_markets_from_event_payload(
                event_payload,
                limit=limit,
            )
            if event_markets:
                self._clear_event_page_backoff(event_slug)
                return [
                    self._parse_market(
                        event_market,
                        market_interval_minutes=market_interval_minutes,
                    )
                    for event_market in event_markets
                ]

        market = self._find_market_by_slug(page_data, event_slug)
        if market is None:
            logger.warning("No market payload found in event page for slug=%s", event_slug)
            return []

        self._clear_event_page_backoff(event_slug)
        return [
            self._parse_market(
                market,
                market_interval_minutes=market_interval_minutes,
            )
        ]

    def _fetch_market_from_event_api_fallback(
        self,
        event_slug: str,
        market_interval_minutes: Optional[int] = None,
    ) -> Optional[MarketInfo]:
        """Fetch event metadata from Gamma API and parse the best child market."""
        markets = self._fetch_markets_from_event_api_fallback(
            event_slug,
            market_interval_minutes=market_interval_minutes,
            limit=1,
        )
        return markets[0] if markets else None

    def _fetch_markets_from_event_api_fallback(
        self,
        event_slug: str,
        market_interval_minutes: Optional[int] = None,
        limit: int = 1,
    ) -> list[MarketInfo]:
        """Fetch event metadata from Gamma API and parse ranked child markets."""
        if limit <= 0:
            return []

        event_payload = self._fetch_event_by_slug_api(event_slug)
        if not event_payload:
            return []

        event_markets = self._select_markets_from_event_payload(
            event_payload,
            limit=limit,
        )
        if not event_markets:
            return []

        return [
            self._parse_market(
                event_market,
                market_interval_minutes=market_interval_minutes,
            )
            for event_market in event_markets
        ]

    def _fetch_btc_5m_market(self) -> Optional[MarketInfo]:
        """
        Query the Gamma API for the best currently-active BTC market.

        Preference order:
        1. Direct BTC 5-minute up/down event pages (`btc-updown-5m-*`).
        2. Legacy BTC 5-minute contract, if it exists in Gamma listings.
        3. Active BTC hourly recurring event, selecting the strike closest to 50/50.
        """
        direct_market = self._fetch_btc_updown_5m_market()
        if direct_market is not None:
            return direct_market

        candidates = self._fetch_btc_5m_market_candidates(limit=1)
        return candidates[0] if candidates else None

    def _fetch_btc_5m_market_candidates(self, limit: int = 10) -> list[MarketInfo]:
        """Return ordered BTC candidates across compatible short-term families."""
        combined_candidates: list[MarketInfo] = []
        combined_candidates.extend(self._fetch_btc_updown_5m_market_candidates())
        combined_candidates.extend(
            self._fetch_listed_btc_5m_market_candidates(limit=max(limit, 5))
        )
        combined_candidates.extend(
            self._fetch_btc_hourly_market_candidates(limit=max(min(limit, 6), 2))
        )

        return self._dedupe_market_candidates(combined_candidates)[:limit]

    def _fetch_listed_btc_5m_market_candidates(
        self,
        limit: int = 5,
    ) -> list[MarketInfo]:
        """Return BTC 5-minute candidates surfaced by generic Gamma listings."""
        # Strategy 1: Search by tag/keyword
        markets = self._search_markets("btc 5 min")
        if not markets:
            # Strategy 2: Broader search
            markets = self._search_markets("bitcoin 5-minute")
        if not markets:
            # Strategy 3: Get all active, filter client-side
            markets = self._get_all_active_markets()
            markets = self._filter_btc_5m(markets)

        if markets:
            active_markets = [
                market
                for market in markets
                if market.get("active", False) and not market.get("closed", True)
            ]
            if not active_markets:
                active_markets = list(markets)

            active_markets.sort(
                key=lambda market: (
                    market.get("end_date_iso", market.get("endDate", market.get("end_date", ""))),
                    str(market.get("slug", "")),
                )
            )
            return [
                self._parse_market(market, market_interval_minutes=5)
                for market in active_markets[:limit]
            ]

        return []

    def _fetch_btc_updown_5m_market(
        self,
        now_ts: Optional[float] = None,
    ) -> Optional[MarketInfo]:
        """
        Resolve the live recurring BTC 5-minute up/down market family.

        Polymarket exposes these markets as event pages whose slugs follow a
        stable pattern:

        `btc-updown-5m-<window_start_epoch_seconds>`

        The page payloads are directly fetchable even when the generic Gamma
        listings do not surface the family reliably, so we probe the current
        window plus its immediate neighbors and choose the market whose 5-minute
        window best matches the current time.
        """
        candidates = self._fetch_btc_updown_5m_market_candidates(now_ts=now_ts)
        return candidates[0] if candidates else None

    def _fetch_btc_updown_5m_market_candidates(
        self,
        now_ts: Optional[float] = None,
    ) -> list[MarketInfo]:
        """Return ordered candidates from the recurring BTC up/down 5-minute family."""
        now_ts = time.time() if now_ts is None else now_ts
        candidates: list[tuple[float, float, MarketInfo]] = []

        for start_ts in self._candidate_btc_updown_5m_start_times(now_ts):
            slug = self._btc_updown_5m_event_slug(start_ts)
            market = self._fetch_market_from_event_slug(
                slug,
                market_interval_minutes=5,
            )
            if market is None:
                continue
            if not market.yes_token_id or not market.no_token_id:
                continue

            end_ts = self._parse_iso_timestamp(market.end_date)
            if end_ts is None:
                end_ts = start_ts + self._btc_updown_interval_seconds
            candidates.append((start_ts, end_ts, market))

        if not candidates:
            return []

        in_window = [
            item for item in candidates if item[0] <= now_ts < item[1]
        ]
        future = [item for item in candidates if item[1] > now_ts and item not in in_window]
        past = [item for item in candidates if item[1] <= now_ts]

        in_window.sort(key=lambda item: (abs(item[0] - now_ts), item[1]))
        future.sort(key=lambda item: (item[0], item[1]))
        past.sort(key=lambda item: item[1], reverse=True)

        ordered_markets: list[MarketInfo] = []
        seen_keys: set[str] = set()
        for _start_ts, _end_ts, market in [*in_window, *future, *past]:
            key = market.condition_id or market.slug
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered_markets.append(market)

        return ordered_markets

    @staticmethod
    def _dedupe_market_candidates(candidates: list[MarketInfo]) -> list[MarketInfo]:
        """Preserve order while removing duplicate candidate markets."""
        ordered_markets: list[MarketInfo] = []
        seen_keys: set[str] = set()
        for market in candidates:
            key = market.condition_id or market.slug
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            ordered_markets.append(market)
        return ordered_markets

    def _candidate_btc_updown_5m_start_times(self, now_ts: float) -> list[int]:
        """Return nearby 5-minute window starts for the recurring up/down family."""
        base_start = int(now_ts // self._btc_updown_interval_seconds) * self._btc_updown_interval_seconds
        starts: list[int] = []
        for offset in (0, -300, 300, -600, 600):
            start_ts = base_start + offset
            if start_ts <= 0 or start_ts in starts:
                continue
            starts.append(start_ts)
        return starts

    @staticmethod
    def _btc_updown_5m_event_slug(start_ts: int) -> str:
        """Build the event-page slug for a BTC 5-minute up/down market."""
        return f"btc-updown-5m-{int(start_ts)}"

    def _search_markets(self, query: str) -> list:
        """Search markets via the Gamma API."""
        try:
            url = f"{self._base_url}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": 20,
            }
            # The Gamma API may support a search/filter mechanism
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            markets = resp.json()

            if isinstance(markets, list):
                return [
                    m for m in markets
                    if self._is_btc_5m_market(m)
                ]
            return []
        except Exception as e:
            logger.debug("Search failed for '%s': %s", query, e)
            return []

    def _get_all_active_markets(self) -> list:
        """Fetch all active markets from Gamma API."""
        results: list[dict] = []
        page_size = 200
        offset = 0
        max_pages = 5
        seen_page_markers: set[tuple[str, ...]] = set()

        try:
            for _ in range(max_pages):
                url = f"{self._base_url}/markets"
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": offset,
                }
                resp = self._session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list) or not payload:
                    break

                page_marker = tuple(
                    str(
                        item.get("id")
                        or item.get("conditionId")
                        or item.get("condition_id")
                        or item.get("slug", "")
                    )
                    for item in payload[:10]
                    if isinstance(item, dict)
                )
                if page_marker in seen_page_markers:
                    break
                seen_page_markers.add(page_marker)

                results.extend(item for item in payload if isinstance(item, dict))
                if len(payload) < page_size:
                    break

                offset += page_size
            return results
        except Exception as e:
            logger.error("Failed to fetch all active markets: %s", e)
            return results

    def _get_active_series(self) -> list:
        """Fetch active recurring series from Gamma API."""
        results: list[dict] = []
        page_size = 200
        offset = 0
        max_pages = 5
        seen_page_markers: set[tuple[str, ...]] = set()

        try:
            for _ in range(max_pages):
                url = f"{self._base_url}/series"
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": offset,
                }
                resp = self._session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list) or not payload:
                    break

                page_marker = tuple(
                    str(item.get("id") or item.get("slug", ""))
                    for item in payload[:10]
                    if isinstance(item, dict)
                )
                if page_marker in seen_page_markers:
                    break
                seen_page_markers.add(page_marker)

                results.extend(item for item in payload if isinstance(item, dict))
                if len(payload) < page_size:
                    break

                offset += page_size
            return results
        except Exception as e:
            logger.error("Failed to fetch active series: %s", e)
            return results

    def get_active_btc_market_candidates(self, limit: int = 10) -> list[dict]:
        """
        Return active BTC market candidates for diagnostics when the short-dated
        strategy cannot find a compatible market.
        """
        if limit <= 0:
            return []

        candidates: list[dict] = []
        for market in self._get_all_active_markets():
            if not self._is_btc_market(market):
                continue

            candidates.append(
                {
                    "slug": str(market.get("slug", "")),
                    "question": str(
                        market.get("question", market.get("title", ""))
                    ),
                    "end_date": str(
                        market.get(
                            "endDate",
                            market.get("end_date_iso", market.get("end_date", "")),
                        )
                    ),
                    "best_bid": self._coerce_float(market.get("bestBid")),
                    "best_ask": self._coerce_float(market.get("bestAsk")),
                }
            )

        candidates.sort(
            key=lambda item: (
                self._parse_iso_timestamp(item["end_date"]) or float("inf"),
                item["slug"],
            )
        )
        return candidates[:limit]

    def _fetch_btc_hourly_market(self) -> Optional[MarketInfo]:
        """
        Fallback for Polymarket's live recurring hourly BTC series.

        The current live structure is an event page containing many threshold
        child markets (for example, "Bitcoin above 65,400 on ..."). We choose
        the active event nearest expiry and then the child market nearest a
        50/50 yes-price so it best approximates an up/down-style binary.
        """
        candidates = self._fetch_btc_hourly_market_candidates(limit=1)
        return candidates[0] if candidates else None

    def _fetch_btc_hourly_market_candidates(
        self,
        limit: int = 4,
    ) -> list[MarketInfo]:
        """Return ranked candidates from the live BTC hourly family."""
        if limit <= 0:
            return []

        series_list = self._get_active_series()
        if not series_list:
            return []

        hourly_series = [
            series for series in series_list if self._is_bitcoin_hourly_series(series)
        ]
        if not hourly_series:
            return []

        preferred = next(
            (
                series
                for series in hourly_series
                if series.get("slug") == "bitcoin-multi-strikes-hourly"
            ),
            hourly_series[0],
        )
        events = self._select_best_series_events(
            preferred.get("events", []),
            limit=2,
        )
        if not events:
            logger.warning(
                "No active BTC hourly event found in series=%s",
                preferred.get("slug"),
            )
            return []

        per_event_limit = max(1, min(3, limit))
        candidates: list[MarketInfo] = []
        for event in events:
            logger.info(
                "Fetched BTC hourly candidate family | event_slug=%s end=%s limit=%d",
                event.get("slug"),
                event.get("endDate"),
                per_event_limit,
            )
            candidates.extend(
                self._fetch_markets_from_event_slug(
                    event["slug"],
                    market_interval_minutes=60,
                    limit=per_event_limit,
                )
            )
            if len(candidates) >= limit:
                break

        return self._dedupe_market_candidates(candidates)[:limit]

    def _fetch_event_by_slug_api(self, event_slug: str) -> Optional[dict]:
        """Fetch event metadata directly from the Gamma API by slug."""
        try:
            url = f"{self._base_url}/events"
            params = {"slug": event_slug}
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            events = resp.json()
            if isinstance(events, list) and events:
                return events[0]
            if isinstance(events, dict) and events:
                return events
            return None
        except Exception as e:
            logger.error("Failed to fetch event %s from API: %s", event_slug, e)
            return None

    def _is_event_page_fetch_backed_off(self, event_slug: str) -> bool:
        """Return True when recent failures mean we should skip page fetches."""
        return self._event_page_backoff_until_by_slug.get(event_slug, 0.0) > time.time()

    def _back_off_event_page_fetch(self, event_slug: str):
        """Pause repeated event-page fetch attempts after transient failures."""
        self._event_page_backoff_until_by_slug[event_slug] = (
            time.time() + self._event_page_backoff_seconds
        )

    def _clear_event_page_backoff(self, event_slug: str):
        """Resume normal event-page fetches after a successful page parse."""
        self._event_page_backoff_until_by_slug.pop(event_slug, None)

    def _filter_btc_5m(self, markets: list) -> list:
        """Filter a list of markets to only BTC 5-minute ones."""
        return [m for m in markets if self._is_btc_5m_market(m)]

    @staticmethod
    def _extract_next_data(html: str) -> Optional[dict[str, Any]]:
        """Extract the embedded Next.js page payload from a Polymarket event page."""
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>',
            html,
            re.DOTALL,
        )
        if not match:
            return None

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_event_payload(payload: dict[str, Any], event_slug: str) -> Optional[dict]:
        """
        Extract an event payload from the hydrated query cache in a Next.js page.
        """
        queries = (
            payload.get("props", {})
            .get("pageProps", {})
            .get("dehydratedState", {})
            .get("queries", [])
        )
        for query in queries:
            query_key = query.get("queryKey")
            if query_key == ["/api/event/slug", event_slug]:
                data = query.get("state", {}).get("data")
                if isinstance(data, dict):
                    return data
        return None

    def _find_market_by_slug(self, payload: Any, event_slug: str) -> Optional[dict]:
        """Walk an embedded page payload and return the first matching market object."""
        if isinstance(payload, dict):
            if payload.get("slug") == event_slug:
                if payload.get("conditionId") or payload.get("condition_id"):
                    return payload

            for value in payload.values():
                market = self._find_market_by_slug(value, event_slug)
                if market is not None:
                    return market

        if isinstance(payload, list):
            for item in payload:
                market = self._find_market_by_slug(item, event_slug)
                if market is not None:
                    return market

        return None

    @staticmethod
    def _is_btc_market(market: dict) -> bool:
        """Check if a market dict is Bitcoin-related at all."""
        searchable_text = " ".join([
            market.get("question", ""),
            market.get("title", ""),
            market.get("slug", ""),
            market.get("description", ""),
        ]).lower()

        return any(kw in searchable_text for kw in ("btc", "bitcoin"))

    @staticmethod
    def _is_btc_5m_market(market: dict) -> bool:
        """
        Check if a market dict represents a BTC 5-minute resolution market.
        Matches on title, question, or slug containing relevant keywords.
        """
        searchable_text = " ".join([
            market.get("question", ""),
            market.get("title", ""),
            market.get("slug", ""),
            market.get("description", ""),
        ]).lower()

        has_btc = GammaAPIClient._is_btc_market(market)
        has_5m = any(kw in searchable_text for kw in ("5 min", "5-min", "5min", "five min"))

        return has_btc and has_5m

    @staticmethod
    def _is_bitcoin_hourly_series(series: dict) -> bool:
        """Identify the current recurring Bitcoin hourly market family."""
        searchable_text = " ".join(
            [
                str(series.get("title", "")),
                str(series.get("slug", "")),
                str(series.get("ticker", "")),
                str(series.get("description", "")),
            ]
        ).lower()
        return (
            ("bitcoin" in searchable_text or "btc" in searchable_text)
            and "hourly" in searchable_text
        )

    @classmethod
    def _select_best_series_event(
        cls,
        events: list[dict],
        now_ts: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Pick the active hourly event closest to expiry but not already stale.
        """
        now_ts = time.time() if now_ts is None else now_ts
        candidates: list[tuple[float, dict]] = []

        for event in events:
            if not event.get("active", False) or event.get("closed", True):
                continue

            end_ts = cls._parse_iso_timestamp(event.get("endDate"))
            if end_ts is None:
                continue
            candidates.append((end_ts, event))

        selected_events = cls._select_best_series_events(
            events,
            now_ts=now_ts,
            limit=1,
        )
        return selected_events[0] if selected_events else None

    @classmethod
    def _select_best_series_events(
        cls,
        events: list[dict],
        now_ts: Optional[float] = None,
        limit: int = 1,
    ) -> list[dict]:
        """Return active series events ordered by nearest relevant expiry."""
        if limit <= 0:
            return []

        now_ts = time.time() if now_ts is None else now_ts
        candidates: list[tuple[float, dict]] = []

        for event in events:
            if not event.get("active", False) or event.get("closed", True):
                continue

            end_ts = cls._parse_iso_timestamp(event.get("endDate"))
            if end_ts is None:
                continue
            candidates.append((end_ts, event))

        if not candidates:
            return []

        future = [item for item in candidates if item[0] >= now_ts]
        if future:
            future.sort(key=lambda item: item[0])
            return [event for _, event in future[:limit]]

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [event for _, event in candidates[:limit]]

    @classmethod
    def _select_market_from_event_payload(cls, event_payload: dict) -> Optional[dict]:
        """
        Select the child market from an event payload that best matches an
        up/down-style binary while preferring the tightest healthy book first.
        """
        markets = cls._select_markets_from_event_payload(event_payload, limit=1)
        return markets[0] if markets else None

    @classmethod
    def _select_markets_from_event_payload(
        cls,
        event_payload: dict,
        limit: int = 1,
    ) -> list[dict]:
        """
        Select one or more ranked child markets from an event payload.

        The ranking prioritizes:
        1. Healthy books over pathological children.
        2. Tighter spreads.
        3. Prices closer to the middle band, which preserves entry headroom.
        4. Higher liquidity as a final tie-breaker.
        """
        if limit <= 0:
            return []

        markets = event_payload.get("markets", [])
        if not isinstance(markets, list) or not markets:
            return []

        scored_markets: list[tuple[float, float, float, str, dict]] = []
        healthy_scored_markets: list[tuple[float, float, float, str, dict]] = []
        centered_healthy_scored_markets: list[tuple[float, float, float, str, dict]] = []
        for market in markets:
            if not market.get("active", False) or market.get("closed", True):
                continue

            yes_price = cls._extract_yes_price(market)
            if yes_price is None:
                continue

            quote_health = cls._event_market_quote_health(market)
            liquidity = cls._coerce_float(
                market.get("liquidityClob", market.get("liquidityNum", market.get("liquidity", 0)))
            )
            score = (
                quote_health["spread"],
                abs(yes_price - 0.5),
                -liquidity,
                str(market.get("slug", "")),
                market,
            )
            scored_markets.append(score)
            if not quote_health["pathological"]:
                healthy_scored_markets.append(score)
                if 0.15 <= yes_price <= 0.85:
                    centered_healthy_scored_markets.append(score)

        selection_pool = (
            centered_healthy_scored_markets
            or healthy_scored_markets
            or scored_markets
        )
        if not selection_pool:
            return []

        selection_pool.sort(key=lambda item: item[:4])
        return [item[4] for item in selection_pool[:limit]]

    @classmethod
    def _event_market_quote_health(cls, market: dict) -> dict[str, float | bool]:
        """Classify child-market quote quality before selecting a series strike."""
        best_bid = cls._coerce_float(market.get("bestBid"))
        best_ask = cls._coerce_float(market.get("bestAsk"))
        has_quotes = best_bid > 0 and best_ask > 0
        spread = (best_ask - best_bid) if has_quotes else float("inf")
        at_price_rails = has_quotes and best_bid <= 0.05 and best_ask >= 0.95
        extreme_spread = has_quotes and spread >= 0.80

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "pathological": (not has_quotes) or at_price_rails or extreme_spread,
        }

    @classmethod
    def _extract_yes_price(cls, market: dict) -> Optional[float]:
        """Extract a normalized yes-side price from a market payload."""
        best_bid = cls._coerce_float(market.get("bestBid"))
        best_ask = cls._coerce_float(market.get("bestAsk"))
        if best_bid > 0 and best_ask > 0:
            return (best_bid + best_ask) / 2.0

        outcome_prices = market.get("outcomePrices")
        if isinstance(outcome_prices, list) and outcome_prices:
            yes_price = cls._coerce_float(outcome_prices[0])
            if yes_price > 0:
                return yes_price

        last_trade_price = cls._coerce_float(market.get("lastTradePrice"))
        return last_trade_price if last_trade_price > 0 else None

    @staticmethod
    def _coerce_float(value: object) -> float:
        """Best-effort float coercion for Gamma payload fields."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_list(value: object) -> list:
        """Normalize list-like Gamma fields that sometimes arrive as JSON strings."""
        if isinstance(value, list):
            return value

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [value]
            return parsed if isinstance(parsed, list) else [parsed]

        return []

    @staticmethod
    def _parse_iso_timestamp(value: Optional[str]) -> Optional[float]:
        """Parse an ISO timestamp into epoch seconds."""
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    @staticmethod
    def _parse_market(
        market: dict,
        market_interval_minutes: Optional[int] = None,
    ) -> MarketInfo:
        """Parse a Gamma API market response into MarketInfo."""
        tokens = GammaAPIClient._coerce_list(market.get("tokens"))
        clob_token_ids = [
            str(token_id)
            for token_id in GammaAPIClient._coerce_list(market.get("clobTokenIds"))
        ]
        outcomes = [
            str(outcome).lower()
            for outcome in GammaAPIClient._coerce_list(market.get("outcomes"))
        ]
        outcome_prices = GammaAPIClient._coerce_list(market.get("outcomePrices"))
        
        yes_token = ""
        no_token = ""
        
        for token in tokens:
            outcome = token.get("outcome", "").lower()
            token_id = token.get("token_id", "")
            if outcome == "yes":
                yes_token = token_id
            elif outcome == "no":
                no_token = token_id

        # Fallback: if tokens aren't labeled, use positional
        if not yes_token and len(tokens) >= 1:
            yes_token = tokens[0].get("token_id", "")
        if not no_token and len(tokens) >= 2:
            no_token = tokens[1].get("token_id", "")

        # Event pages often expose token IDs as clobTokenIds rather than nested tokens.
        if not yes_token and len(clob_token_ids) >= 1:
            yes_token = clob_token_ids[0]
        if not no_token and len(clob_token_ids) >= 2:
            no_token = clob_token_ids[1]

        # Preserve expected ordering for Up/Down style markets.
        if len(clob_token_ids) >= 2 and len(outcomes) >= 2:
            if outcomes[0] in {"up", "yes"} and outcomes[1] in {"down", "no"}:
                yes_token = clob_token_ids[0]
                no_token = clob_token_ids[1]

        indicative_yes = None
        indicative_no = None
        if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
            indicative_yes = GammaAPIClient._coerce_float(outcome_prices[0])
            indicative_no = GammaAPIClient._coerce_float(outcome_prices[1])

        return MarketInfo(
            condition_id=market.get(
                "condition_id",
                market.get("conditionId", market.get("id", "")),
            ),
            question=market.get("question", market.get("title", "")),
            slug=market.get("slug", ""),
            yes_token_id=yes_token,
            no_token_id=no_token,
            end_date=market.get(
                "end_date_iso",
                market.get("endDate", market.get("end_date", "")),
            ),
            active=market.get("active", True),
            neg_risk=market.get("neg_risk", market.get("negRisk", False)),
            indicative_yes_price=indicative_yes,
            indicative_no_price=indicative_no,
            min_order_size=GammaAPIClient._coerce_float(
                market.get("orderMinSize", market.get("order_min_size"))
            )
            or None,
            market_interval_minutes=market_interval_minutes,
        )

    def get_market_by_id(self, market_id: str) -> Optional[dict]:
        """Fetch a specific market by its condition ID."""
        try:
            url = f"{self._base_url}/markets/{market_id}"
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Failed to fetch market %s: %s", market_id, e)
            return None

    def close(self):
        """Close the HTTP session."""
        self._session.close()
