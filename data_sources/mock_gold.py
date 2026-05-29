"""Mock market data for the first version of the gold evaluator.

No external API is called here. All values are deterministic so tests can assert
stable results.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from data_sources.models import GoldMarketSnapshot


def get_mock_gold_snapshot() -> GoldMarketSnapshot:
    """Return deterministic mock data for the current gold evaluation."""

    return GoldMarketSnapshot(
        price=2368.4,
        dxy=104.2,
        real_yield_10y=1.84,
        fed_funds_rate=5.33,
        etf_flow_tonnes_5d=3.7,
        geopolitical_risk_index=62.0,
        as_of=datetime(2026, 5, 29, 0, 0, tzinfo=timezone.utc),
    )


def get_mock_gold_price_history() -> pd.DataFrame:
    """Return deterministic daily mock OHLC data for technical indicators."""

    dates = pd.date_range(end="2026-05-29", periods=80, freq="D", tz="UTC")
    close = pd.Series(
        [
            2230,
            2228,
            2236,
            2244,
            2240,
            2248,
            2252,
            2257,
            2254,
            2262,
            2269,
            2266,
            2275,
            2283,
            2288,
            2285,
            2294,
            2301,
            2298,
            2306,
            2312,
            2310,
            2318,
            2327,
            2332,
            2329,
            2338,
            2344,
            2341,
            2350,
            2358,
            2354,
            2362,
            2370,
            2366,
            2373,
            2381,
            2376,
            2384,
            2390,
            2386,
            2394,
            2401,
            2396,
            2404,
            2410,
            2405,
            2412,
            2418,
            2414,
            2421,
            2428,
            2423,
            2416,
            2408,
            2401,
            2394,
            2388,
            2381,
            2376,
            2371,
            2367,
            2363,
            2358,
            2355,
            2359,
            2362,
            2365,
            2368,
            2371,
            2375,
            2372,
            2370,
            2369,
            2366,
            2364,
            2367,
            2370,
            2368,
            2368.4,
        ],
        dtype="float64",
    )
    high = close + 12.0
    low = close - 11.0
    open_ = close.shift(1).fillna(close.iloc[0] - 3.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
        }
    )
