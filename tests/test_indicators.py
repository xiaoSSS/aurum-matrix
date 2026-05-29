from __future__ import annotations

import pandas as pd

from data_sources.mock_gold import get_mock_gold_price_history
from indicators.technical import calculate_technical_indicators, moving_average


def test_moving_average_calculates_expected_value() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = moving_average(series, 3)

    assert result.iloc[-1] == 3.0


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
