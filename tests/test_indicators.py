from __future__ import annotations

import math

import pandas as pd
import pytest

from data_sources.mock_gold import get_mock_gold_price_history
from indicators.technical import (
    atr,
    calculate_technical_indicators,
    ema,
    macd,
    moving_average,
    rsi,
    simple_moving_average,
)


def test_simple_moving_average_returns_latest_value_and_series() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = simple_moving_average(series, 3)

    assert result.latest_value == 3.0
    assert result.series.iloc[-1] == 3.0
    assert math.isnan(result.series.iloc[0])


def test_moving_average_backward_compatible_series() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = moving_average(series, 3)

    assert result.iloc[-1] == 3.0


def test_ema_returns_latest_value_and_series() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = ema(series, 3)

    assert result.latest_value == result.series.iloc[-1]
    assert len(result.series) == len(series)
    assert result.latest_value > 0


def test_macd_returns_latest_values_and_full_series() -> None:
    close = pd.Series([float(value) for value in range(1, 60)])

    result = macd(close)

    assert result.latest_macd == result.macd.iloc[-1]
    assert result.latest_signal == result.signal.iloc[-1]
    assert result.latest_histogram == result.histogram.iloc[-1]
    assert len(result.macd) == len(close)
    assert len(result.signal) == len(close)
    assert len(result.histogram) == len(close)


def test_rsi_returns_latest_value_and_series() -> None:
    close = pd.Series(
        [
            44.0,
            44.5,
            45.0,
            44.2,
            44.8,
            45.4,
            46.0,
            45.6,
            46.2,
            46.8,
            47.1,
            46.9,
            47.4,
            47.8,
            48.1,
        ]
    )

    result = rsi(close, 14)

    assert result.latest_value == result.series.iloc[-1]
    assert 0 <= result.latest_value <= 100
    assert len(result.series) == len(close)


def test_atr_returns_latest_value_and_series() -> None:
    high = pd.Series([11.0, 12.0, 13.0, 14.0, 15.0])
    low = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0])
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])

    result = atr(high, low, close, 3)

    assert result.latest_value == result.series.iloc[-1]
    assert result.latest_value == 2.0
    assert len(result.series) == len(close)


@pytest.mark.parametrize(
    ("indicator", "args"),
    [
        (simple_moving_average, (pd.Series(dtype=float), 3)),
        (ema, (pd.Series([1.0, 2.0]), 3)),
        (rsi, (pd.Series([1.0, float("nan"), 3.0, 4.0]), 3)),
        (macd, (pd.Series([1.0] * 20),)),
        (
            atr,
            (
                pd.Series([2.0, float("nan"), 4.0]),
                pd.Series([1.0, 2.0, 3.0]),
                pd.Series([1.5, 2.5, 3.5]),
                2,
            ),
        ),
    ],
)
def test_indicators_raise_for_empty_insufficient_or_nan_data(indicator, args) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        indicator(*args)


def test_atr_raises_for_mismatched_indexes() -> None:
    high = pd.Series([3.0, 4.0, 5.0], index=[0, 1, 2])
    low = pd.Series([2.0, 3.0, 4.0], index=[0, 1, 3])
    close = pd.Series([2.5, 3.5, 4.5], index=[0, 1, 2])

    with pytest.raises(ValueError, match="matching indexes"):
        atr(high, low, close, 2)


def test_calculate_technical_indicators_from_mock_history() -> None:
    history = get_mock_gold_price_history()

    indicators = calculate_technical_indicators(history)

    assert indicators is not None
    assert indicators.ma_short > 0
    assert indicators.ma_long > 0
    assert 0 <= indicators.rsi <= 100
    assert indicators.atr > 0


def test_calculate_technical_indicators_returns_none_for_missing_data() -> None:
    history = get_mock_gold_price_history().head(10)

    indicators = calculate_technical_indicators(history)

    assert indicators is None
