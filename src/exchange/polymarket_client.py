"""
Polymarket CLOB client wrapper.
Handles authentication, order placement (post-only GTD), cancellation, and balance queries.
Uses py-clob-client under the hood with proper Proxy/Safe signature support.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    OrderArgs,
    OrderType,
)
from py_clob_client.order_builder.constants import BUY, SELL

from config.settings import POLYMARKET, TRADING

logger = logging.getLogger(__name__)
COLLATERAL_DECIMALS = 6


@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[dict] = None


@dataclass
class BalanceAllowanceStatus:
    """Normalized balance/allowance view for a Polymarket asset."""
    balance: float
    allowance: float
    allowances_by_spender: Optional[dict[str, float]] = None
    raw_response: Optional[dict] = None

    @property
    def available_to_trade(self) -> float:
        """Effective buying power is capped by the lower of balance and allowance."""
        return min(self.balance, self.allowance)


class PolymarketClient:
    """
    High-level wrapper around py-clob-client for Polymarket CLOB operations.
    Configured for gasless execution via Relayer (proxy wallet).
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        funder_address: Optional[str] = None,
        signature_type: Optional[int] = None,
    ):
        self._private_key = (
            private_key if private_key is not None else POLYMARKET.private_key
        )
        self._funder = (
            funder_address
            if funder_address is not None
            else POLYMARKET.funder_address
        )
        self._sig_type = (
            signature_type
            if signature_type is not None
            else POLYMARKET.signature_type
        )

        # Build credentials
        creds = None
        _api_key = api_key if api_key is not None else POLYMARKET.api_key
        _api_secret = api_secret if api_secret is not None else POLYMARKET.api_secret
        _api_passphrase = (
            api_passphrase
            if api_passphrase is not None
            else POLYMARKET.api_passphrase
        )

        if _api_key and _api_secret and _api_passphrase:
            creds = ApiCreds(
                api_key=_api_key,
                api_secret=_api_secret,
                api_passphrase=_api_passphrase,
            )

        self._client = ClobClient(
            host=POLYMARKET.clob_host,
            key=self._private_key or None,
            chain_id=POLYMARKET.chain_id,
            creds=creds,
            signature_type=self._sig_type,
            funder=self._funder,
        )
        self._data_session = requests.Session()
        self._data_session.headers.update({
            "Accept": "application/json",
            "User-Agent": "polymarket-btc-bot/0.1",
        })

        logger.info(
            "PolymarketClient initialized | sig_type=%d funder=%s",
            self._sig_type,
            self._funder[:10] + "..." if self._funder else "N/A",
        )

    @property
    def has_signing_key(self) -> bool:
        """Whether a private key is available for level-1 authenticated calls."""
        return self._client.signer is not None

    @property
    def has_trading_access(self) -> bool:
        """Whether the client can place/cancel orders and fetch private data."""
        return self.has_signing_key and self._client.creds is not None

    @property
    def tracking_address(self) -> Optional[str]:
        """
        Public address used for Data API position/P&L lookups.

        Polymarket's Data API expects the profile/proxy wallet address. When a
        funder/proxy address is configured, prefer that. Otherwise fall back to
        the signer's address for standard EOA setups.
        """
        if self._funder:
            return self._funder

        try:
            return self._client.get_address()
        except Exception:
            return None

    def derive_api_creds(self) -> ApiCreds:
        """
        Derive L2 HMAC-SHA256 credentials from the private key.
        Call this once when setting up a new wallet.
        """
        if not self.has_signing_key:
            raise ValueError("Cannot derive API credentials without a private key")
        creds = self._client.create_or_derive_api_creds()
        logger.info("Derived API credentials: key=%s", creds.api_key[:8] + "...")
        return creds

    def set_api_creds(self, creds: ApiCreds):
        """Inject API credentials into the client."""
        self._client.set_api_creds(creds)

    def place_post_only_gtd(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        ttl_seconds: int = None,
        neg_risk: bool = False,
    ) -> OrderResult:
        """
        Place a post-only GTD limit order.
        
        Args:
            token_id: The conditional token ID (Yes or No outcome).
            price: Limit price (0.01 - 0.99).
            size: Number of shares (default: 1).
            ttl_seconds: Time-to-live before expiration. 
            side: "BUY" or "SELL".
            neg_risk: Whether this is a negative-risk market.
            
        Returns:
            OrderResult with success status and order ID.
            
        CRITICAL: post_only=True is MANDATORY to guarantee Maker status
        and avoid the taker fee which would destroy our micro-capital edge.
        """
        if not self.has_trading_access:
            return OrderResult(
                success=False,
                error="Trading credentials unavailable",
            )

        ttl = ttl_seconds or TRADING.gtd_ttl_seconds
        expiration = int(time.time()) + ttl + 60  # Buffer for CLOB 10s security threshold

        order_side = BUY if side.upper() == "BUY" else SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=order_side,
            expiration=expiration,
        )

        try:
            # Step 1: Create and sign the order
            signed_order = self._client.create_order(order_args)

            # Step 2: Post with post_only=True and GTD type
            response = self._client.post_order(
                signed_order,
                OrderType.GTD,
                post_only=True,  # post_only — NEVER change this
            )

            order_id = response.get("orderID", response.get("id", "unknown"))
            logger.info(
                "Order placed | id=%s side=%s price=%.2f size=%.1f token=%s ttl=%ds",
                order_id, side, price, size, token_id[:12] + "...", ttl,
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                raw_response=response,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error("Order placement failed: %s", error_msg)
            return OrderResult(success=False, error=error_msg)

    def cancel_all_orders(self) -> bool:
        """
        Cancel all resting limit orders. 
        Used by the risk manager during volatility kill-switch events.
        """
        if not self.has_trading_access:
            logger.info("Skipping cancel_all_orders: trading credentials unavailable")
            return False
        try:
            response = self._client.cancel_all()
            logger.warning("MASS CANCEL executed | response=%s", response)
            return True
        except Exception as e:
            logger.error("Mass cancel failed: %s", e)
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order by ID."""
        if not self.has_trading_access:
            logger.info("Skipping cancel_order(%s): trading credentials unavailable", order_id)
            return False
        try:
            self._client.cancel(order_id)
            logger.info("Order cancelled | id=%s", order_id)
            return True
        except Exception as e:
            logger.error("Cancel failed for %s: %s", order_id, e)
            return False

    def get_open_orders(self) -> list:
        """Retrieve all currently open/resting orders."""
        if not self.has_trading_access:
            return []
        try:
            orders = self._client.get_orders()
            return orders if orders else []
        except Exception as e:
            logger.error("Failed to fetch open orders: %s", e)
            return []

    def get_balance_allowance(
        self,
        asset_type: AssetType,
        token_id: Optional[str] = None,
    ) -> Optional[BalanceAllowanceStatus]:
        """
        Fetch and normalize balance/allowance information for an asset.

        Polymarket documents `balance` and `allowance` as string fields, but
        this wrapper parses them defensively so router code can fail closed.
        """
        if not self.has_trading_access:
            logger.info(
                "Skipping balance check: trading credentials unavailable"
            )
            return None

        params = BalanceAllowanceParams(
            asset_type=asset_type,
            token_id=token_id,
            signature_type=self._sig_type,
        )

        try:
            response = self._client.get_balance_allowance(params)
            raw_balance = self._extract_raw_field(response, "balance")
            raw_allowances = self._extract_allowances(response)
            raw_allowance = self._extract_raw_field(response, "allowance")
            collateral_base_units = (
                asset_type == AssetType.COLLATERAL
                and (
                    self._looks_like_base_units(raw_balance, COLLATERAL_DECIMALS)
                    or self._looks_like_base_units(raw_allowance, COLLATERAL_DECIMALS)
                    or any(
                        self._looks_like_base_units(value, COLLATERAL_DECIMALS)
                        for value in raw_allowances.values()
                    )
                )
            )
            balance = self._normalize_balance_allowance(
                raw_balance,
                asset_type,
                force_base_units=collateral_base_units,
            )
            allowances_by_spender = {
                spender: self._normalize_balance_allowance(
                    value,
                    asset_type,
                    force_base_units=collateral_base_units,
                )
                for spender, value in raw_allowances.items()
            }
            allowance = self._normalize_balance_allowance(
                raw_allowance,
                asset_type,
                force_base_units=collateral_base_units,
            )
            if allowances_by_spender:
                # The live endpoint can return a spender->allowance map instead
                # of a single top-level allowance field.
                allowance = max(allowance, max(allowances_by_spender.values()))

            return BalanceAllowanceStatus(
                balance=balance,
                allowance=allowance,
                allowances_by_spender=allowances_by_spender or None,
                raw_response=response if isinstance(response, dict) else None,
            )
        except Exception as e:
            logger.error(
                "Failed to fetch balance/allowance for asset_type=%s token=%s: %s",
                asset_type,
                token_id[:12] + "..." if token_id else "N/A",
                e,
            )
            return None

    def get_collateral_balance_allowance(self) -> Optional[BalanceAllowanceStatus]:
        """Fetch USDC collateral balance/allowance for buy-side order checks."""
        return self.get_balance_allowance(AssetType.COLLATERAL)

    def get_available_collateral(self) -> Optional[float]:
        """Return the currently tradeable collateral after allowance capping."""
        status = self.get_collateral_balance_allowance()
        if status is None:
            return None
        return status.available_to_trade

    def get_current_positions(
        self,
        limit: int = 50,
        max_records: int = 200,
    ) -> list[dict]:
        """Fetch current positions from the public Polymarket Data API."""
        return self._get_data_api_positions(
            endpoint="/positions",
            limit=limit,
            max_records=max_records,
        )

    def get_closed_positions(
        self,
        limit: int = 50,
        max_records: int = 200,
    ) -> list[dict]:
        """Fetch closed positions from the public Polymarket Data API."""
        return self._get_data_api_positions(
            endpoint="/closed-positions",
            limit=limit,
            max_records=max_records,
        )

    def has_sufficient_collateral(self, required_amount: float) -> bool:
        """
        Return True when both collateral balance and allowance cover the order cost.
        """
        if required_amount <= 0:
            return True

        status = self.get_collateral_balance_allowance()
        if status is None:
            logger.error(
                "Unable to verify collateral balance/allowance; blocking order"
            )
            return False

        if status.available_to_trade + 1e-9 < required_amount:
            logger.warning(
                "Insufficient collateral/allowance | required=%.4f balance=%.4f "
                "allowance=%.4f available=%.4f",
                required_amount,
                status.balance,
                status.allowance,
                status.available_to_trade,
            )
            return False

        return True

    def get_order_book(self, token_id: str):
        """
        Fetch the current order book for a specific token.
        Returns the raw client response, which may be a dict-like payload
        or a typed OrderBookSummary object.
        """
        try:
            book = self._client.get_order_book(token_id)
            return book
        except Exception as e:
            logger.error("Failed to fetch order book for %s: %s", token_id[:12], e)
            return {"bids": [], "asks": []}

    def get_best_bid_ask(self, token_id: str) -> tuple[Optional[float], Optional[float]]:
        """
        Get the best bid and ask prices for a token.
        Returns (best_bid, best_ask) or (None, None) on failure.
        """
        book = self.get_order_book(token_id)
        
        best_bid = None
        best_ask = None

        if isinstance(book, dict):
            bids = book.get("bids", [])
            asks = book.get("asks", [])
        else:
            bids = getattr(book, "bids", [])
            asks = getattr(book, "asks", [])

        def _price(level) -> float:
            raw_price = level.get("price", 0) if isinstance(level, dict) else getattr(level, "price", 0)
            return float(raw_price)

        if bids:
            best_bid = max(_price(level) for level in bids)
        if asks:
            best_ask = min(_price(level) for level in asks)
        
        return best_bid, best_ask

    def get_trade_history(self, limit: int = 100) -> list:
        """Fetch recent trade history for the authenticated user."""
        if not self.has_trading_access:
            return []
        try:
            trades = self._client.get_trades()
            if not trades:
                return []
            return trades[:limit]
        except Exception as e:
            logger.error("Failed to fetch trade history: %s", e)
            return []

    def _get_data_api_positions(
        self,
        endpoint: str,
        limit: int,
        max_records: int,
    ) -> list[dict]:
        """Page through a public Data API positions endpoint."""
        user = self.tracking_address
        if not user:
            logger.info(
                "Skipping %s fetch: no funder or signer address is available",
                endpoint,
            )
            return []

        if limit <= 0 or max_records <= 0:
            return []

        results: list[dict] = []
        page_size = max(1, min(limit, 50))
        offset = 0

        while len(results) < max_records:
            batch_limit = min(page_size, max_records - len(results))
            try:
                response = self._data_session.get(
                    f"{POLYMARKET.data_api_base}{endpoint}",
                    params={
                        "user": user,
                        "limit": batch_limit,
                        "offset": offset,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as e:
                logger.error(
                    "Failed to fetch %s for %s: %s",
                    endpoint,
                    user[:10] + "...",
                    e,
                )
                break

            if not isinstance(payload, list):
                logger.error(
                    "Unexpected %s payload type: %s",
                    endpoint,
                    type(payload).__name__,
                )
                break

            results.extend(item for item in payload if isinstance(item, dict))
            if len(payload) < batch_limit:
                break

            offset += batch_limit

        return results

    @staticmethod
    def _extract_numeric_field(payload: object, field: str) -> float:
        """Extract numeric values from dict-like or object responses."""
        return PolymarketClient._coerce_float(
            PolymarketClient._extract_raw_field(payload, field)
        )

    @staticmethod
    def _extract_raw_field(payload: object, field: str) -> object:
        """Extract raw field values from dict-like or object responses."""
        if isinstance(payload, dict):
            if field in payload:
                return payload[field]
            nested = payload.get("data")
            if isinstance(nested, dict) and field in nested:
                return nested[field]

        return getattr(payload, field, 0.0)

    @staticmethod
    def _extract_allowances(payload: object) -> dict[str, object]:
        """Extract a spender->allowance map from dict-like responses."""
        if isinstance(payload, dict):
            allowances = payload.get("allowances")
            if isinstance(allowances, dict):
                return allowances

            nested = payload.get("data")
            if isinstance(nested, dict):
                nested_allowances = nested.get("allowances")
                if isinstance(nested_allowances, dict):
                    return nested_allowances

        return {}

    @staticmethod
    def _normalize_balance_allowance(
        value: object,
        asset_type: AssetType,
        force_base_units: bool = False,
    ) -> float:
        """
        Normalize balance/allowance fields into human-readable token units.

        The live collateral endpoint has been observed returning integer-like
        USDC.e base units, while tests and some SDK examples use decimal strings.
        We only scale integer-like collateral amounts that look like 6-decimal
        base units.
        """
        amount = PolymarketClient._coerce_float(value)
        if asset_type != AssetType.COLLATERAL:
            return amount
        if force_base_units and PolymarketClient._looks_integer_like(value):
            return amount / (10**COLLATERAL_DECIMALS)
        if not PolymarketClient._looks_like_base_units(value, COLLATERAL_DECIMALS):
            return amount
        return amount / (10**COLLATERAL_DECIMALS)

    @staticmethod
    def _looks_like_base_units(value: object, decimals: int) -> bool:
        """Heuristically detect integer-like onchain base-unit amounts."""
        threshold = 10**decimals

        if isinstance(value, int):
            return abs(value) >= threshold

        if isinstance(value, float):
            return value.is_integer() and abs(value) >= threshold

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return False
            if stripped[0] in "+-":
                stripped = stripped[1:]
            if not stripped or any(ch in stripped for ch in ".eE"):
                return False
            if not stripped.isdigit():
                return False
            return int(stripped) >= threshold

        return False

    @staticmethod
    def _looks_integer_like(value: object) -> bool:
        """Return True when a value is an integer-ish numeric representation."""
        if isinstance(value, int):
            return True

        if isinstance(value, float):
            return value.is_integer()

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return False
            if stripped[0] in "+-":
                stripped = stripped[1:]
            return stripped.isdigit()

        return False

    @staticmethod
    def _coerce_float(value: object) -> float:
        """Parse numeric strings defensively, returning 0.0 on malformed input."""
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0

        if isinstance(value, dict):
            for key in ("value", "amount", "decimal"):
                if key in value:
                    return PolymarketClient._coerce_float(value[key])

        return 0.0
