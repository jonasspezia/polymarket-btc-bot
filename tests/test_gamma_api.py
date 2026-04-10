import json
from types import SimpleNamespace

from src.exchange.gamma_api import GammaAPIClient, MarketInfo


def test_extract_next_data_and_find_market_by_slug():
    payload = {
        "props": {
            "pageProps": {
                "event": {
                    "markets": [
                        {
                            "slug": "bitcoin-up-or-down-april-6-2026-5pm-et",
                            "conditionId": "0xabc",
                            "clobTokenIds": ["up-token", "down-token"],
                            "outcomes": ["Up", "Down"],
                            "question": "Bitcoin Up or Down - April 6, 5PM ET",
                        }
                    ]
                }
            }
        }
    }
    html = (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload)}"
        "</script></body></html>"
    )

    extracted = GammaAPIClient._extract_next_data(html)

    assert extracted == payload

    client = GammaAPIClient()
    market = client._find_market_by_slug(
        extracted,
        "bitcoin-up-or-down-april-6-2026-5pm-et",
    )

    assert market is not None
    assert market["conditionId"] == "0xabc"


def test_parse_market_uses_clob_token_ids_for_up_down_market():
    market = {
        "conditionId": "0xabc",
        "question": "Bitcoin Up or Down - April 6, 5PM ET",
        "slug": "bitcoin-up-or-down-april-6-2026-5pm-et",
        "outcomes": ["Up", "Down"],
        "outcomePrices": ["0.52", "0.48"],
        "clobTokenIds": ["up-token", "down-token"],
        "orderMinSize": "5",
        "endDate": "2026-04-06T22:00:00Z",
        "negRisk": False,
    }

    parsed = GammaAPIClient._parse_market(market)

    assert parsed.condition_id == "0xabc"
    assert parsed.yes_token_id == "up-token"
    assert parsed.no_token_id == "down-token"
    assert parsed.end_date == "2026-04-06T22:00:00Z"
    assert parsed.indicative_yes_price == 0.52
    assert parsed.indicative_no_price == 0.48
    assert parsed.min_order_size == 5.0
    assert parsed.market_interval_minutes is None


def test_parse_market_preserves_interval_metadata():
    market = {
        "conditionId": "0xdef",
        "question": "Bitcoin above 70,200 on April 6, 1PM ET",
        "slug": "bitcoin-above-70200-on-april-6-2026-1pm-et",
        "outcomes": ["Yes", "No"],
        "clobTokenIds": ["yes-token", "no-token"],
        "endDate": "2026-04-06T17:00:00Z",
    }

    parsed = GammaAPIClient._parse_market(market, market_interval_minutes=60)

    assert parsed.market_interval_minutes == 60


def test_parse_market_handles_json_encoded_list_fields():
    market = {
        "conditionId": "0xghi",
        "question": "Bitcoin above 69,200 on April 6, 2PM ET",
        "slug": "bitcoin-above-69200-on-april-6-2026-2pm-et",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.49", "0.51"]',
        "clobTokenIds": '["yes-token", "no-token"]',
        "endDate": "2026-04-06T18:00:00Z",
    }

    parsed = GammaAPIClient._parse_market(market, market_interval_minutes=60)

    assert parsed.yes_token_id == "yes-token"
    assert parsed.no_token_id == "no-token"
    assert parsed.indicative_yes_price == 0.49
    assert parsed.indicative_no_price == 0.51


def test_extract_event_payload_reads_hydrated_query_cache():
    payload = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [
                        {
                            "queryKey": ["/api/event/slug", "bitcoin-above-on-april-5-2026-3pm-et"],
                            "state": {
                                "data": {
                                    "slug": "bitcoin-above-on-april-5-2026-3pm-et",
                                    "markets": [{"slug": "child-market"}],
                                }
                            },
                        }
                    ]
                }
            }
        }
    }

    extracted = GammaAPIClient._extract_event_payload(
        payload,
        "bitcoin-above-on-april-5-2026-3pm-et",
    )

    assert extracted is not None
    assert extracted["slug"] == "bitcoin-above-on-april-5-2026-3pm-et"
    assert extracted["markets"][0]["slug"] == "child-market"


def test_select_market_from_event_payload_prefers_most_balanced_market():
    event_payload = {
        "markets": [
            {
                "slug": "deep-itm",
                "active": True,
                "closed": False,
                "bestBid": 0.95,
                "bestAsk": 0.97,
                "liquidityClob": 100.0,
            },
            {
                "slug": "balanced",
                "active": True,
                "closed": False,
                "bestBid": 0.49,
                "bestAsk": 0.51,
                "liquidityClob": 50.0,
            },
            {
                "slug": "deep-otm",
                "active": True,
                "closed": False,
                "bestBid": 0.03,
                "bestAsk": 0.05,
                "liquidityClob": 200.0,
            },
        ]
    }

    selected = GammaAPIClient._select_market_from_event_payload(event_payload)

    assert selected is not None
    assert selected["slug"] == "balanced"


def test_select_market_from_event_payload_prefers_healthy_quotes_over_rails():
    event_payload = {
        "markets": [
            {
                "slug": "balanced-but-pathological",
                "active": True,
                "closed": False,
                "bestBid": 0.05,
                "bestAsk": 0.95,
                "liquidityClob": 1000.0,
            },
            {
                "slug": "healthy-alternative",
                "active": True,
                "closed": False,
                "bestBid": 0.44,
                "bestAsk": 0.48,
                "liquidityClob": 25.0,
            },
        ]
    }

    selected = GammaAPIClient._select_market_from_event_payload(event_payload)

    assert selected is not None
    assert selected["slug"] == "healthy-alternative"


def test_select_market_from_event_payload_prefers_tighter_healthy_spread_over_balance():
    event_payload = {
        "markets": [
            {
                "slug": "more-balanced-but-wide",
                "active": True,
                "closed": False,
                "bestBid": 0.47,
                "bestAsk": 0.67,
                "liquidityClob": 200.0,
            },
            {
                "slug": "less-balanced-but-tight",
                "active": True,
                "closed": False,
                "bestBid": 0.39,
                "bestAsk": 0.43,
                "liquidityClob": 20.0,
            },
        ]
    }

    selected = GammaAPIClient._select_market_from_event_payload(event_payload)

    assert selected is not None
    assert selected["slug"] == "less-balanced-but-tight"


def test_select_markets_from_event_payload_returns_ranked_candidates():
    event_payload = {
        "markets": [
            {
                "slug": "widish",
                "active": True,
                "closed": False,
                "bestBid": 0.44,
                "bestAsk": 0.50,
                "liquidityClob": 30.0,
            },
            {
                "slug": "tight-center",
                "active": True,
                "closed": False,
                "bestBid": 0.48,
                "bestAsk": 0.51,
                "liquidityClob": 10.0,
            },
            {
                "slug": "tight-off-center",
                "active": True,
                "closed": False,
                "bestBid": 0.34,
                "bestAsk": 0.37,
                "liquidityClob": 50.0,
            },
        ]
    }

    selected = GammaAPIClient._select_markets_from_event_payload(
        event_payload,
        limit=2,
    )

    assert [market["slug"] for market in selected] == [
        "tight-off-center",
        "tight-center",
    ]


def test_select_market_from_event_payload_falls_back_when_all_children_are_pathological():
    event_payload = {
        "markets": [
            {
                "slug": "missing-quotes",
                "active": True,
                "closed": False,
                "bestBid": 0.0,
                "bestAsk": 0.0,
                "outcomePrices": ["0.49", "0.51"],
                "liquidityClob": 20.0,
            },
            {
                "slug": "rails",
                "active": True,
                "closed": False,
                "bestBid": 0.05,
                "bestAsk": 0.95,
                "liquidityClob": 10.0,
            },
        ]
    }

    selected = GammaAPIClient._select_market_from_event_payload(event_payload)

    assert selected is not None
    assert selected["slug"] == "rails"


def test_select_best_series_event_prefers_nearest_future_end_time():
    events = [
        {
            "slug": "past",
            "active": True,
            "closed": False,
            "endDate": "2026-04-05T17:00:00Z",
        },
        {
            "slug": "future-near",
            "active": True,
            "closed": False,
            "endDate": "2026-04-05T19:00:00Z",
        },
        {
            "slug": "future-far",
            "active": True,
            "closed": False,
            "endDate": "2026-04-05T21:00:00Z",
        },
    ]

    selected = GammaAPIClient._select_best_series_event(
        events,
        now_ts=GammaAPIClient._parse_iso_timestamp("2026-04-05T18:30:00Z"),
    )

    assert selected is not None
    assert selected["slug"] == "future-near"


def test_fetch_market_from_event_slug_falls_back_to_api_when_page_fetch_fails():
    client = GammaAPIClient()

    def raise_on_get(*_args, **_kwargs):
        raise ConnectionResetError("connection reset")

    client._session = SimpleNamespace(get=raise_on_get)
    client._fetch_event_by_slug_api = lambda slug: {
        "slug": slug,
        "markets": [
            {
                "slug": "balanced",
                "active": True,
                "closed": False,
                "bestBid": 0.49,
                "bestAsk": 0.51,
                "liquidityClob": 50.0,
                "conditionId": "0xabc",
                "question": "Bitcoin above 70,200 on April 6, 2PM ET",
                "outcomes": ["Yes", "No"],
                "clobTokenIds": ["yes-token", "no-token"],
                "endDate": "2026-04-06T18:00:00Z",
            }
        ],
    }

    market = client._fetch_market_from_event_slug(
        "bitcoin-above-on-april-6-2026-2pm-et",
        market_interval_minutes=60,
    )

    assert market is not None
    assert market.slug == "balanced"
    assert market.market_interval_minutes == 60


def test_fetch_market_from_event_slug_skips_page_fetch_during_backoff():
    client = GammaAPIClient()
    calls = {"page": 0, "api": 0}

    def raise_on_get(*_args, **_kwargs):
        calls["page"] += 1
        raise ConnectionResetError("connection reset")

    def fetch_event_payload(slug):
        calls["api"] += 1
        return {
            "slug": slug,
            "markets": [
                {
                    "slug": "balanced",
                    "active": True,
                    "closed": False,
                    "bestBid": 0.49,
                    "bestAsk": 0.51,
                    "liquidityClob": 50.0,
                    "conditionId": "0xabc",
                    "question": "Bitcoin above 70,200 on April 6, 2PM ET",
                    "outcomes": ["Yes", "No"],
                    "clobTokenIds": ["yes-token", "no-token"],
                    "endDate": "2026-04-06T18:00:00Z",
                }
            ],
        }

    client._session = SimpleNamespace(get=raise_on_get)
    client._fetch_event_by_slug_api = fetch_event_payload

    first = client._fetch_market_from_event_slug(
        "bitcoin-above-on-april-6-2026-2pm-et",
        market_interval_minutes=60,
    )
    second = client._fetch_market_from_event_slug(
        "bitcoin-above-on-april-6-2026-2pm-et",
        market_interval_minutes=60,
    )

    assert first is not None
    assert second is not None
    assert calls["page"] == 1
    assert calls["api"] == 2


def test_btc_updown_5m_event_slug_uses_window_start_timestamp():
    assert GammaAPIClient._btc_updown_5m_event_slug(1775587500) == "btc-updown-5m-1775587500"


def test_candidate_btc_updown_5m_start_times_center_on_current_window():
    client = GammaAPIClient()

    starts = client._candidate_btc_updown_5m_start_times(1775587968.0)

    assert starts == [
        1775587800,
        1775587500,
        1775588100,
        1775587200,
        1775588400,
    ]


def test_fetch_btc_updown_5m_market_prefers_window_containing_now():
    client = GammaAPIClient()
    now_ts = 1775587968.0

    def fake_fetch(slug, market_interval_minutes=None):
        start_ts = int(slug.rsplit("-", 1)[-1])
        return MarketInfo(
            condition_id=f"condition-{start_ts}",
            question=f"Bitcoin Up or Down - window {start_ts}",
            slug=slug,
            yes_token_id=f"yes-{start_ts}",
            no_token_id=f"no-{start_ts}",
            end_date="2026-04-07T18:45:00Z" if start_ts == 1775587200 else
            "2026-04-07T18:50:00Z" if start_ts == 1775587500 else
            "2026-04-07T18:55:00Z" if start_ts == 1775587800 else
            "2026-04-07T19:00:00Z" if start_ts == 1775588100 else
            "2026-04-07T19:05:00Z",
            market_interval_minutes=market_interval_minutes,
        )

    client._fetch_market_from_event_slug = fake_fetch

    market = client._fetch_btc_updown_5m_market(now_ts=now_ts)

    assert market is not None
    assert market.slug == "btc-updown-5m-1775587800"
    assert market.market_interval_minutes == 5


def test_fetch_btc_updown_5m_market_candidates_returns_ordered_neighbors():
    client = GammaAPIClient()
    now_ts = 1775587968.0

    def fake_fetch(slug, market_interval_minutes=None):
        start_ts = int(slug.rsplit("-", 1)[-1])
        return MarketInfo(
            condition_id=f"condition-{start_ts}",
            question=f"Bitcoin Up or Down - window {start_ts}",
            slug=slug,
            yes_token_id=f"yes-{start_ts}",
            no_token_id=f"no-{start_ts}",
            end_date="2026-04-07T18:45:00Z" if start_ts == 1775587200 else
            "2026-04-07T18:50:00Z" if start_ts == 1775587500 else
            "2026-04-07T18:55:00Z" if start_ts == 1775587800 else
            "2026-04-07T19:00:00Z" if start_ts == 1775588100 else
            "2026-04-07T19:05:00Z",
            market_interval_minutes=market_interval_minutes,
        )

    client._fetch_market_from_event_slug = fake_fetch

    candidates = client._fetch_btc_updown_5m_market_candidates(now_ts=now_ts)

    assert [candidate.slug for candidate in candidates] == [
        "btc-updown-5m-1775587800",
        "btc-updown-5m-1775588100",
        "btc-updown-5m-1775588400",
        "btc-updown-5m-1775587500",
        "btc-updown-5m-1775587200",
    ]


def test_get_active_btc_5m_market_candidates_prefers_direct_updown_family():
    client = GammaAPIClient()
    sentinel_current = MarketInfo(
        condition_id="0x5m-current",
        question="Current",
        slug="btc-updown-5m-current",
        yes_token_id="yes-current",
        no_token_id="no-current",
        end_date="2026-04-07T18:55:00Z",
        market_interval_minutes=5,
    )
    sentinel_next = MarketInfo(
        condition_id="0x5m-next",
        question="Next",
        slug="btc-updown-5m-next",
        yes_token_id="yes-next",
        no_token_id="no-next",
        end_date="2026-04-07T19:00:00Z",
        market_interval_minutes=5,
    )
    listed = MarketInfo(
        condition_id="0xlisted",
        question="Listed",
        slug="bitcoin-up-or-down-listed",
        yes_token_id="yes-listed",
        no_token_id="no-listed",
        end_date="2026-04-07T19:05:00Z",
        market_interval_minutes=5,
    )
    hourly_a = MarketInfo(
        condition_id="0xhourly-a",
        question="Hourly A",
        slug="bitcoin-multi-strikes-hourly-child-a",
        yes_token_id="yes-hourly-a",
        no_token_id="no-hourly-a",
        end_date="2026-04-07T20:00:00Z",
        market_interval_minutes=60,
    )
    hourly_b = MarketInfo(
        condition_id="0xhourly-b",
        question="Hourly B",
        slug="bitcoin-multi-strikes-hourly-child-b",
        yes_token_id="yes-hourly-b",
        no_token_id="no-hourly-b",
        end_date="2026-04-07T21:00:00Z",
        market_interval_minutes=60,
    )

    client._fetch_btc_updown_5m_market_candidates = lambda: [
        sentinel_current,
        sentinel_next,
    ]
    client._fetch_listed_btc_5m_market_candidates = lambda limit=5: [listed]
    client._fetch_btc_hourly_market_candidates = lambda limit=4: [
        hourly_a,
        hourly_b,
    ]

    candidates = client.get_active_btc_5m_market_candidates(force_refresh=True)

    assert candidates == [
        sentinel_current,
        sentinel_next,
        listed,
        hourly_a,
        hourly_b,
    ]


def test_fetch_btc_hourly_market_candidates_returns_ranked_children_from_nearest_events():
    client = GammaAPIClient()
    client._get_active_series = lambda: [
        {
            "slug": "bitcoin-multi-strikes-hourly",
            "title": "Bitcoin Multi Strikes Hourly",
            "events": [
                {
                    "slug": "event-near",
                    "active": True,
                    "closed": False,
                    "endDate": "2099-04-07T19:00:00Z",
                },
                {
                    "slug": "event-next",
                    "active": True,
                    "closed": False,
                    "endDate": "2099-04-07T20:00:00Z",
                },
            ],
        }
    ]

    def fake_fetch(event_slug, market_interval_minutes=None, limit=1):
        if event_slug == "event-near":
            return [
                MarketInfo(
                    condition_id="0xnear-a",
                    question="Near A",
                    slug="near-a",
                    yes_token_id="yes-near-a",
                    no_token_id="no-near-a",
                    end_date="2026-04-07T19:00:00Z",
                    market_interval_minutes=market_interval_minutes,
                ),
                MarketInfo(
                    condition_id="0xnear-b",
                    question="Near B",
                    slug="near-b",
                    yes_token_id="yes-near-b",
                    no_token_id="no-near-b",
                    end_date="2026-04-07T19:00:00Z",
                    market_interval_minutes=market_interval_minutes,
                ),
            ][:limit]
        return [
            MarketInfo(
                condition_id="0xnext-a",
                question="Next A",
                slug="next-a",
                yes_token_id="yes-next-a",
                no_token_id="no-next-a",
                end_date="2026-04-07T20:00:00Z",
                market_interval_minutes=market_interval_minutes,
            )
        ][:limit]

    client._fetch_markets_from_event_slug = fake_fetch

    selected = client._fetch_btc_hourly_market_candidates(limit=3)

    assert [market.slug for market in selected] == ["near-a", "near-b", "next-a"]


def test_fetch_btc_5m_market_prefers_direct_updown_family_before_hourly_fallback():
    client = GammaAPIClient()
    sentinel = MarketInfo(
        condition_id="0x5m",
        question="Bitcoin Up or Down - April 7, 2:50PM-2:55PM ET",
        slug="btc-updown-5m-1775587800",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2026-04-07T18:55:00Z",
        market_interval_minutes=5,
    )

    client._fetch_btc_updown_5m_market = lambda: sentinel
    client._search_markets = lambda _query: (_ for _ in ()).throw(AssertionError("search should not run"))
    client._get_all_active_markets = lambda: (_ for _ in ()).throw(AssertionError("active market scan should not run"))
    client._fetch_btc_hourly_market = lambda: (_ for _ in ()).throw(AssertionError("hourly fallback should not run"))

    market = client._fetch_btc_5m_market()

    assert market == sentinel
