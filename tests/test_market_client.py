from __future__ import annotations

import sqlite3

import pytest

from data_sources.market_client import (
    ALPHA_VANTAGE_QUERY_URL,
    AlphaVantageMarketClient,
    MarketApiKeyError,
    MarketDataProvider,
    MarketRequestError,
    MarketResponseError,
)

TEST_ALPHA_VANTAGE_API_KEY = "demo-alpha-key"


def _xau_payload() -> dict[str, object]:
    return {
        "Meta Data": {"1. Information": "Forex Daily Prices"},
        "Time Series FX (Daily)": {
            "2026-05-29": {
                "1. open": "2360.10",
                "2. high": "2375.20",
                "3. low": "2351.30",
                "4. close": "2368.40",
            },
            "2026-05-28": {
                "1. open": "2350.00",
                "2. high": "2365.00",
                "3. low": "2345.00",
                "4. close": "2360.10",
            },
        },
    }


def _dxy_payload() -> dict[str, object]:
    return {
        "Meta Data": {"1. Information": "Daily Prices"},
        "Time Series (Daily)": {
            "2026-05-29": {
                "1. open": "104.10",
                "2. high": "104.30",
                "3. low": "103.90",
                "4. close": "104.20",
                "5. volume": "123456",
            }
        },
    }


def test_alpha_vantage_client_reads_api_key_from_environment(monkeypatch, requests_mock) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", TEST_ALPHA_VANTAGE_API_KEY)
    requests_mock.get(ALPHA_VANTAGE_QUERY_URL, json=_xau_payload())

    client = AlphaVantageMarketClient()
    data = client.get_xau_usd_daily(outputsize="compact")

    assert isinstance(client, MarketDataProvider)
    assert data.symbol == "XAU/USD"
    assert data.provider == "alpha_vantage"
    assert data.cached is False
    assert [bar.date for bar in data.bars] == ["2026-05-28", "2026-05-29"]
    assert data.bars[-1].open == 2360.10
    assert data.bars[-1].high == 2375.20
    assert data.bars[-1].low == 2351.30
    assert data.bars[-1].close == 2368.40
    assert data.bars[-1].volume is None
    request = requests_mock.last_request
    assert request is not None
    assert request.qs["function"] == ["fx_daily"]
    assert request.qs["from_symbol"] == ["xau"]
    assert request.qs["to_symbol"] == ["usd"]
    assert request.qs["apikey"] == [TEST_ALPHA_VANTAGE_API_KEY]


def test_alpha_vantage_client_raises_when_api_key_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    with pytest.raises(MarketApiKeyError, match="ALPHA_VANTAGE_API_KEY"):
        AlphaVantageMarketClient()


def test_alpha_vantage_client_gets_dxy_daily_with_unified_fields(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(ALPHA_VANTAGE_QUERY_URL, json=_dxy_payload())
    client = AlphaVantageMarketClient(api_key=TEST_ALPHA_VANTAGE_API_KEY)

    data = client.get_dxy_daily(outputsize="full")

    assert data.symbol == "DXY"
    assert data.bars[0].date == "2026-05-29"
    assert data.bars[0].open == 104.10
    assert data.bars[0].high == 104.30
    assert data.bars[0].low == 103.90
    assert data.bars[0].close == 104.20
    assert data.bars[0].volume == 123456
    request = requests_mock.last_request
    assert request is not None
    assert request.qs["function"] == ["time_series_daily"]
    assert request.qs["symbol"] == ["dxy"]
    assert request.qs["outputsize"] == ["full"]


def test_alpha_vantage_client_routes_get_daily_by_symbol(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(ALPHA_VANTAGE_QUERY_URL, json=_dxy_payload())
    client = AlphaVantageMarketClient(api_key=TEST_ALPHA_VANTAGE_API_KEY)

    data = client.get_daily("DXY")

    assert data.symbol == "DXY"


def test_alpha_vantage_client_caches_daily_data_to_sqlite(tmp_path, requests_mock) -> None:  # type: ignore[no-untyped-def]
    cache_path = tmp_path / "market_cache.db"
    requests_mock.get(ALPHA_VANTAGE_QUERY_URL, json=_xau_payload())
    client = AlphaVantageMarketClient(
        api_key=TEST_ALPHA_VANTAGE_API_KEY,
        cache_path=cache_path,
    )

    first = client.get_xau_usd_daily()
    second = client.get_xau_usd_daily()

    assert first.cached is False
    assert second.cached is True
    assert second.bars[-1].close == 2368.40
    assert requests_mock.call_count == 1
    with sqlite3.connect(cache_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM market_data_cache").fetchone()[
            0
        ]
    assert count == 1


def test_alpha_vantage_client_raises_clear_error_on_http_failure(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(ALPHA_VANTAGE_QUERY_URL, status_code=503, text="unavailable")
    client = AlphaVantageMarketClient(api_key=TEST_ALPHA_VANTAGE_API_KEY)

    with pytest.raises(MarketRequestError, match="XAU/USD"):
        client.get_xau_usd_daily()


def test_alpha_vantage_client_raises_clear_error_on_provider_error(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(
        ALPHA_VANTAGE_QUERY_URL, json={"Error Message": "Invalid API call."}
    )
    client = AlphaVantageMarketClient(api_key=TEST_ALPHA_VANTAGE_API_KEY)

    with pytest.raises(MarketResponseError, match="Invalid API call"):
        client.get_xau_usd_daily()


def test_alpha_vantage_client_raises_clear_error_on_missing_time_series(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(ALPHA_VANTAGE_QUERY_URL, json={"Meta Data": {}})
    client = AlphaVantageMarketClient(api_key=TEST_ALPHA_VANTAGE_API_KEY)

    with pytest.raises(MarketResponseError, match="Time Series FX"):
        client.get_xau_usd_daily()


def test_alpha_vantage_client_raises_clear_error_on_invalid_numeric_field(requests_mock) -> None:  # type: ignore[no-untyped-def]
    payload = _xau_payload()
    payload["Time Series FX (Daily)"]["2026-05-29"]["4. close"] = "not-a-number"  # type: ignore[index]
    requests_mock.get(ALPHA_VANTAGE_QUERY_URL, json=payload)
    client = AlphaVantageMarketClient(api_key=TEST_ALPHA_VANTAGE_API_KEY)

    with pytest.raises(MarketResponseError, match="4. close"):
        client.get_xau_usd_daily()
