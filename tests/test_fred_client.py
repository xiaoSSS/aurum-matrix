from __future__ import annotations

import sqlite3

import pytest

from data_sources.fred_client import (
    FRED_OBSERVATIONS_URL,
    TEN_YEAR_BREAKEVEN_INFLATION_SERIES_ID,
    TEN_YEAR_REAL_RATE_SERIES_ID,
    TEN_YEAR_TREASURY_YIELD_SERIES_ID,
    FredApiKeyError,
    FredClient,
    FredRequestError,
    FredResponseError,
)

TEST_FRED_API_KEY = "42641757dfd839995f55a9abd3b8ba04"


def _fred_payload(value: str = "1.75") -> dict[str, object]:
    return {
        "observations": [
            {
                "realtime_start": "2026-05-29",
                "realtime_end": "2026-05-29",
                "date": "2026-05-28",
                "value": value,
            }
        ]
    }


def test_fred_client_reads_api_key_from_environment(monkeypatch, requests_mock) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRED_API_KEY", TEST_FRED_API_KEY)
    requests_mock.get(FRED_OBSERVATIONS_URL, json=_fred_payload("1.84"))

    client = FredClient()
    data = client.get_series_history("DFII10", observation_start="2026-01-01")

    assert data.series_id == "DFII10"
    assert data.observations[0].value == 1.84
    assert data.cached is False
    request = requests_mock.last_request
    assert request is not None
    assert request.qs["api_key"] == [TEST_FRED_API_KEY]
    assert request.qs["series_id"] == ["dfii10"]
    assert request.qs["observation_start"] == ["2026-01-01"]


def test_fred_client_raises_when_api_key_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(FredApiKeyError, match="FRED_API_KEY"):
        FredClient()


def test_fred_client_caches_series_history_to_sqlite(tmp_path, requests_mock) -> None:  # type: ignore[no-untyped-def]
    cache_path = tmp_path / "fred_cache.db"
    requests_mock.get(FRED_OBSERVATIONS_URL, json=_fred_payload("4.50"))
    client = FredClient(api_key=TEST_FRED_API_KEY, cache_path=cache_path)

    first = client.get_series_history(
        "DGS10", observation_start="2026-01-01", observation_end="2026-01-31"
    )
    second = client.get_series_history(
        "DGS10", observation_start="2026-01-01", observation_end="2026-01-31"
    )

    assert first.cached is False
    assert second.cached is True
    assert second.observations[0].value == 4.5
    assert requests_mock.call_count == 1
    with sqlite3.connect(cache_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM fred_series_cache").fetchone()[
            0
        ]
    assert count == 1


def test_fred_client_supports_required_macro_series(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(FRED_OBSERVATIONS_URL, json=_fred_payload("2.20"))
    client = FredClient(api_key=TEST_FRED_API_KEY)

    real_rate = client.get_10y_real_rate()
    treasury_yield = client.get_10y_treasury_yield()
    inflation_expectations = client.get_inflation_expectations()

    assert real_rate.series_id == TEN_YEAR_REAL_RATE_SERIES_ID
    assert treasury_yield.series_id == TEN_YEAR_TREASURY_YIELD_SERIES_ID
    assert inflation_expectations.series_id == TEN_YEAR_BREAKEVEN_INFLATION_SERIES_ID
    assert requests_mock.call_count == 3


def test_fred_client_preserves_missing_observation_values(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(FRED_OBSERVATIONS_URL, json=_fred_payload("."))
    client = FredClient(api_key=TEST_FRED_API_KEY)

    data = client.get_series_history("DFII10")

    assert data.observations[0].value is None


def test_fred_client_raises_clear_error_on_http_failure(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(FRED_OBSERVATIONS_URL, status_code=500, text="server error")
    client = FredClient(api_key=TEST_FRED_API_KEY)

    with pytest.raises(FredRequestError, match="DFII10"):
        client.get_series_history("DFII10")


def test_fred_client_raises_clear_error_on_fred_error_payload(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(
        FRED_OBSERVATIONS_URL,
        json={"error_code": 400, "error_message": "Bad Request. Invalid series_id."},
    )
    client = FredClient(api_key=TEST_FRED_API_KEY)

    with pytest.raises(FredResponseError, match="Invalid series_id"):
        client.get_series_history("BAD")


def test_fred_client_raises_clear_error_on_malformed_payload(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.get(FRED_OBSERVATIONS_URL, json={"not_observations": []})
    client = FredClient(api_key=TEST_FRED_API_KEY)

    with pytest.raises(FredResponseError, match="missing observations"):
        client.get_series_history("DFII10")
