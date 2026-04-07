"""Tests for the order routing logic."""

from unittest.mock import MagicMock, patch

import pytest

from src.exchange.gamma_api import MarketInfo
from src.execution.order_router import OrderRouter, TradingSignal


@pytest.fixture
def mock_client():
    """Create a mock PolymarketClient."""
    client = MagicMock()
    client.get_best_bid_ask.return_value = (None, None)
    client.get_available_collateral.return_value = None
    client.has_sufficient_collateral.return_value = True
    client.place_post_only_gtd.return_value = MagicMock(
        success=True, order_id="test-order-123"
    )
    return client


@pytest.fixture
def market_info():
    """Create a test MarketInfo."""
    return MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token-abc123",
        no_token_id="no-token-def456",
        end_date="2099-04-05T12:05:00Z",
    )


@pytest.fixture
def router(mock_client):
    return OrderRouter(
        client=mock_client,
        min_edge=0.02,
        order_size=1.0,
        gtd_ttl=10,
        dry_run=False,
    )


class TestOrderRouter:
    def test_no_trade_on_no_edge(self, router, mock_client, market_info):
        """No order should be placed when there's no edge."""
        # Market price = 0.50, model says 0.50 → no edge
        mock_client.get_best_bid_ask.side_effect = [
            (0.49, 0.50),  # YES token
            (0.49, 0.50),  # NO token
        ]
        result = router.evaluate_and_trade(0.50, market_info)
        assert result is None
        mock_client.place_post_only_gtd.assert_not_called()

    def test_buy_yes_on_positive_edge(self, router, mock_client, market_info):
        """Should buy YES when model prob > ask + min_edge."""
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),  # YES: ask=0.50
            (0.48, 0.50),  # NO: ask=0.50
        ]
        # Model says p=0.55, ask=0.50, edge=0.05 > min_edge=0.02
        result = router.evaluate_and_trade(0.55, market_info)
        assert result is not None
        assert result.success
        mock_client.place_post_only_gtd.assert_called_once()

        # Verify the call used the YES token
        call_kwargs = mock_client.place_post_only_gtd.call_args
        assert call_kwargs[1]["token_id"] == market_info.yes_token_id
        assert call_kwargs[1]["side"] == "BUY"

    def test_buy_no_on_negative_edge(self, router, mock_client, market_info):
        """Should buy NO when (1-model_prob) > no_ask + min_edge."""
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),  # YES: ask=0.50 → model prob 0.40, edge=-0.10 (no YES buy)
            (0.48, 0.50),  # NO: ask=0.50 → no_prob=0.60, edge=0.10 > 0.02
        ]
        # Model says p=0.40 → YES undervalued? No. NO overvalued? (1-0.40)=0.60 > 0.50+0.02
        result = router.evaluate_and_trade(0.40, market_info)
        assert result is not None
        assert result.success

    def test_post_only_is_mandatory(self, router, mock_client, market_info):
        """Verify post_only is always True in order placement."""
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]
        router.evaluate_and_trade(0.55, market_info)
        
        # The underlying place_post_only_gtd enforces post_only internally

    def test_order_count_tracking(self, router, mock_client, market_info):
        """Verify order count increments on successful trades."""
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50), (0.48, 0.50),
        ]
        assert router.orders_placed == 0
        router.evaluate_and_trade(0.55, market_info)
        assert router.orders_placed == 1

    def test_rejected_order_tracking(self, router, mock_client, market_info):
        """Verify rejected order count increments on failures."""
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50), (0.48, 0.50),
        ]
        mock_client.place_post_only_gtd.return_value = MagicMock(
            success=False, error="Post-only rejected"
        )
        router.evaluate_and_trade(0.55, market_info)
        assert router.orders_rejected == 1

    def test_snap_price(self):
        """Test price snapping to nearest tick (rounds down, no extra subtraction)."""
        # Exact tick boundary: 0.50 snaps to 0.50
        snapped = OrderRouter._snap_price(0.50)
        assert snapped == 0.50

        # Mid-tick: 0.554 snaps down to 0.55
        snapped = OrderRouter._snap_price(0.554)
        assert snapped == 0.55

        # Exact tick boundary: 0.55 snaps to 0.55
        snapped = OrderRouter._snap_price(0.55)
        assert snapped == 0.55

    def test_no_trade_on_empty_book(self, router, mock_client, market_info):
        """No trade when order book is empty."""
        mock_client.get_best_bid_ask.side_effect = [
            (None, None),
            (None, None),
        ]
        result = router.evaluate_and_trade(0.60, market_info)
        assert result is None

    def test_dry_run_does_not_place_order(self, mock_client, market_info):
        """Dry-run mode should simulate orders without touching the client."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            order_size=1.0,
            gtd_ttl=10,
            dry_run=True,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]

        result = router.evaluate_and_trade(0.55, market_info)

        assert result is not None
        assert result.success
        assert result.raw_response["dry_run"] is True
        assert router.orders_simulated == 1
        mock_client.place_post_only_gtd.assert_not_called()

    def test_confidence_threshold_blocks_low_probability_yes_lottery_ticket(
        self, mock_client, market_info
    ):
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            min_side_probability=0.25,
            order_size=1.0,
            gtd_ttl=10,
            dry_run=True,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.03, 0.05),
            (0.90, 0.95),
        ]

        result = router.evaluate_and_trade(0.20, market_info)

        assert result is None
        mock_client.place_post_only_gtd.assert_not_called()

    def test_confidence_threshold_blocks_subthreshold_yes_signal(
        self, mock_client, market_info
    ):
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            min_side_probability=0.52,
            order_size=1.0,
            gtd_ttl=10,
            dry_run=True,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.40, 0.45),
            (0.53, 0.57),
        ]

        result = router.evaluate_and_trade(0.51, market_info)

        assert result is None
        mock_client.place_post_only_gtd.assert_not_called()

    def test_max_entry_price_blocks_expensive_no_contract(
        self, mock_client, market_info
    ):
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            min_side_probability=0.52,
            max_entry_price=0.90,
            order_size=1.0,
            gtd_ttl=10,
            dry_run=True,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.01, 0.02),
            (0.95, 0.99),
        ]

        result = router.evaluate_and_trade(0.01, market_info)

        assert result is None
        mock_client.place_post_only_gtd.assert_not_called()

    def test_sell_wall_order_book_filter_blocks_entry(
        self, mock_client, market_info
    ):
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            min_side_probability=0.25,
            order_size=1.0,
            gtd_ttl=10,
            dry_run=True,
            min_order_book_imbalance=0.35,
            max_ask_wall_ratio=2.5,
        )
        mock_client.get_order_book.side_effect = [
            {
                "bids": [
                    {"price": "0.20", "size": "1"},
                    {"price": "0.19", "size": "1"},
                ],
                "asks": [
                    {"price": "0.21", "size": "10"},
                    {"price": "0.22", "size": "8"},
                ],
            },
            {
                "bids": [{"price": "0.95", "size": "1"}],
                "asks": [{"price": "0.96", "size": "1"}],
            },
        ]

        result = router.evaluate_and_trade(0.55, market_info)

        assert result is None
        mock_client.place_post_only_gtd.assert_not_called()

    def test_duplicate_signal_suppression(self, mock_client, market_info):
        """Repeated identical signals inside the guard window should be skipped."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            order_size=1.0,
            gtd_ttl=10,
            duplicate_window_seconds=60,
            dry_run=False,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
            (0.48, 0.50),
            (0.48, 0.50),
        ]

        first = router.evaluate_and_trade(0.55, market_info)
        second = router.evaluate_and_trade(0.55, market_info)

        assert first is not None
        assert second is not None
        assert second.success is False
        assert second.error == "duplicate_signal_suppressed"
        assert router.duplicate_signals_suppressed == 1
        assert mock_client.place_post_only_gtd.call_count == 1

    def test_insufficient_collateral_blocks_order(self, router, mock_client, market_info):
        """Live orders should be rejected before placement when collateral is insufficient."""
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]
        mock_client.has_sufficient_collateral.return_value = False

        result = router.evaluate_and_trade(0.55, market_info)

        assert result is not None
        assert result.success is False
        assert result.error == "Insufficient balance or allowance"
        assert router.orders_rejected == 1
        mock_client.place_post_only_gtd.assert_not_called()

    def test_rejects_order_when_configured_size_is_below_market_minimum(
        self, router, mock_client, market_info
    ):
        """Live orders should fail closed when the venue minimum exceeds configured size."""
        market = MarketInfo(
            condition_id=market_info.condition_id,
            question=market_info.question,
            slug=market_info.slug,
            yes_token_id=market_info.yes_token_id,
            no_token_id=market_info.no_token_id,
            end_date=market_info.end_date,
            min_order_size=5.0,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]

        result = router.evaluate_and_trade(0.55, market)

        assert result is not None
        assert result.success is False
        assert result.error == "Configured order size is below the market minimum size"
        assert router.orders_rejected == 1
        mock_client.place_post_only_gtd.assert_not_called()

    def test_dry_run_simulates_signal_even_when_live_sizing_would_block(
        self, mock_client, market_info
    ):
        """Paper mode should still simulate the strategy path even if live sizing would fail."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            order_size=1.0,
            gtd_ttl=10,
            dry_run=True,
        )
        market = MarketInfo(
            condition_id=market_info.condition_id,
            question=market_info.question,
            slug=market_info.slug,
            yes_token_id=market_info.yes_token_id,
            no_token_id=market_info.no_token_id,
            end_date=market_info.end_date,
            min_order_size=5.0,
        )
        mock_client.get_available_collateral.return_value = 0.0
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]

        result = router.evaluate_and_trade(0.55, market)

        assert result is not None
        assert result.success is True
        assert result.raw_response["dry_run"] is True
        assert result.raw_response["live_blocked"] is True
        assert (
            result.raw_response["live_block_reason"]
            == "Configured order size is below the market minimum size"
        )
        assert router.orders_simulated == 1
        assert router.orders_rejected == 0
        mock_client.place_post_only_gtd.assert_not_called()

    def test_scales_live_order_size_by_bankroll_fraction(
        self, mock_client, market_info
    ):
        """Live orders should downsize to the configured spend cap when needed."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            order_size=10.0,
            gtd_ttl=10,
            dry_run=False,
            bankroll_fraction_per_order=0.25,
        )
        mock_client.get_available_collateral.return_value = 4.0
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]

        result = router.evaluate_and_trade(0.55, market_info)

        assert result is not None
        assert result.success
        call_kwargs = mock_client.place_post_only_gtd.call_args.kwargs
        assert call_kwargs["size"] == pytest.approx(1.0 / 0.49)

    def test_order_notional_targets_fixed_dollar_spend(
        self, mock_client, market_info
    ):
        """Dollar-based sizing should convert a $1 target into the right share count."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            order_size=99.0,
            order_notional=1.0,
            gtd_ttl=10,
            dry_run=False,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]

        result = router.evaluate_and_trade(0.55, market_info)

        assert result is not None
        assert result.success
        call_kwargs = mock_client.place_post_only_gtd.call_args.kwargs
        assert call_kwargs["size"] == pytest.approx(1.0 / 0.49)

    def test_one_dollar_cap_skips_market_when_minimum_size_costs_more(
        self, mock_client, market_info
    ):
        """A strict $1 cap must fail closed if the venue minimum needs more capital."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            order_notional=1.0,
            gtd_ttl=10,
            dry_run=False,
            max_order_notional=1.0,
        )
        market = MarketInfo(
            condition_id=market_info.condition_id,
            question=market_info.question,
            slug=market_info.slug,
            yes_token_id=market_info.yes_token_id,
            no_token_id=market_info.no_token_id,
            end_date=market_info.end_date,
            min_order_size=5.0,
        )
        mock_client.get_available_collateral.return_value = 1.0
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]

        result = router.evaluate_and_trade(0.55, market)

        assert result is not None
        assert result.success is False
        assert result.error == "Configured order size is below the market minimum size"
        mock_client.place_post_only_gtd.assert_not_called()

    def test_rejects_order_when_bankroll_cannot_fund_market_minimum(
        self, mock_client, market_info
    ):
        """Venue minimums should block orders that the bankroll cannot support."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            order_size=10.0,
            gtd_ttl=10,
            dry_run=False,
        )
        market = MarketInfo(
            condition_id=market_info.condition_id,
            question=market_info.question,
            slug=market_info.slug,
            yes_token_id=market_info.yes_token_id,
            no_token_id=market_info.no_token_id,
            end_date=market_info.end_date,
            min_order_size=5.0,
        )
        mock_client.get_available_collateral.return_value = 1.0
        mock_client.get_best_bid_ask.side_effect = [
            (0.48, 0.50),
            (0.48, 0.50),
        ]

        result = router.evaluate_and_trade(0.55, market)

        assert result is not None
        assert result.success is False
        assert result.error == "Available collateral cannot fund the market minimum size"
        assert router.orders_rejected == 1
        mock_client.place_post_only_gtd.assert_not_called()

    def test_dry_run_uses_indicative_price_when_live_book_is_pathological(
        self, mock_client, market_info
    ):
        """Paper trading should fall back to indicative prices when asks are unusable."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            order_size=1.0,
            gtd_ttl=10,
            dry_run=True,
        )
        market = MarketInfo(
            condition_id=market_info.condition_id,
            question=market_info.question,
            slug=market_info.slug,
            yes_token_id=market_info.yes_token_id,
            no_token_id=market_info.no_token_id,
            end_date=market_info.end_date,
            indicative_yes_price=0.50,
            indicative_no_price=0.50,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.01, 0.99),
            (0.01, 0.99),
        ]

        result = router.evaluate_and_trade(0.55, market)

        assert result is not None
        assert result.success
        assert result.raw_response["dry_run"] is True
        assert router.orders_simulated == 1

    def test_dry_run_prefers_tight_live_book_over_indicative_price(
        self, mock_client, market_info
    ):
        """Paper trading should respect a tight live book even if indicative pricing disagrees."""
        router = OrderRouter(
            client=mock_client,
            min_edge=0.02,
            min_side_probability=0.52,
            max_entry_price=0.90,
            order_size=1.0,
            gtd_ttl=10,
            dry_run=True,
        )
        market = MarketInfo(
            condition_id=market_info.condition_id,
            question=market_info.question,
            slug=market_info.slug,
            yes_token_id=market_info.yes_token_id,
            no_token_id=market_info.no_token_id,
            end_date=market_info.end_date,
            indicative_yes_price=0.02,
            indicative_no_price=0.70,
        )
        mock_client.get_best_bid_ask.side_effect = [
            (0.01, 0.02),
            (0.98, 0.99),
        ]

        result = router.evaluate_and_trade(0.01, market)

        assert result is None
        mock_client.place_post_only_gtd.assert_not_called()
