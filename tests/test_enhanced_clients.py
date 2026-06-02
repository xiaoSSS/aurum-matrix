from __future__ import annotations

from data_sources.enhanced_clients import (
    EnhancedDataClient,
    cftc_position_state,
    get_latest_value,
    trend_from_observations,
)
from data_sources.sqlite_cache import TimeSeriesCache


def test_enhanced_client_parses_csv_file_and_uses_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "etf.csv"
    source.write_text("date,value\n2026-01-01,1.5\n2026-01-02,2.5\n", encoding="utf-8")
    cache_path = tmp_path / "cache.db"
    client = EnhancedDataClient(
        source="wgc",
        series_id="etf_flows",
        url=str(source),
        cache_path=cache_path,
        cache_ttl_hours=24,
    )
    first = client.get_series()
    second = EnhancedDataClient(
        source="wgc",
        series_id="etf_flows",
        cache_path=cache_path,
        cache_ttl_hours=24,
    ).get_series()
    assert get_latest_value(first) == 2.5
    assert trend_from_observations(first) == "up"
    assert second.cached is True
    assert get_latest_value(second) == 2.5


def test_cftc_position_state_from_cached_series(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = TimeSeriesCache(tmp_path / "cache.db")
    cache.write_points(
        source="cftc",
        series_id="gold_cot",
        points=[
            ("2026-01-01", 100.0, {}),
            ("2026-01-08", 150.0, {}),
        ],
    )
    series = EnhancedDataClient(
        source="cftc",
        series_id="gold_cot",
        cache_path=tmp_path / "cache.db",
    ).get_series()
    assert cftc_position_state(series) == "net_long_increasing"
