from __future__ import annotations

import pytest

from scoring.decision_engine import DecisionThresholds, decide_from_total_score
from scoring.factor_score import (
    FactorScoreConfig,
    FactorScoreInput,
    calculate_factor_score,
)


def _input(**overrides: object) -> FactorScoreInput:
    values: dict[str, object] = {
        "gold_price": 2400.0,
        "ma20": 2380.0,
        "ma60": 2350.0,
        "ma120": 2300.0,
        "macd_value": 12.0,
        "macd_signal": 8.0,
        "rsi": 55.0,
        "real_yield_trend": "down",
        "dxy_trend": "down",
        "etf_flow_trend": "up",
        "cftc_position_state": "net_long_increasing",
    }
    values.update(overrides)
    return FactorScoreInput(**values)


def test_factor_score_strong_bullish_case() -> None:
    result = calculate_factor_score(_input())

    assert result.macro_score == 65
    assert result.technical_score == 65
    assert result.flow_score == 60
    assert result.total_score == 64
    assert result.decision == "强多"
    assert any("实际利率趋势下行" in reason for reason in result.reasons)
    assert result.risks == ["未发现单一因子极端风险，但本结果不构成收益承诺。"]


def test_factor_score_bullish_case() -> None:
    result = calculate_factor_score(
        _input(
            dxy_trend="flat",
            ma20=2350.0,
            ma60=2380.0,
            ma120=2300.0,
            rsi=72.0,
            etf_flow_trend="flat",
        )
    )

    assert result.macro_score == 35
    assert result.technical_score == 30
    assert result.flow_score == 25
    assert result.total_score == 30
    assert result.decision == "偏多"
    assert any("RSI 处于高位" in reason for reason in result.reasons)


def test_factor_score_neutral_case() -> None:
    result = calculate_factor_score(
        _input(
            gold_price=2350.0,
            ma20=2360.0,
            ma60=2340.0,
            ma120=2350.0,
            real_yield_trend="flat",
            dxy_trend="flat",
            etf_flow_trend="flat",
            cftc_position_state="neutral",
        )
    )

    assert result.macro_score == 0
    assert result.technical_score == 20
    assert result.flow_score == 0
    assert result.total_score == 8
    assert result.decision == "中性"


def test_factor_score_bearish_case() -> None:
    result = calculate_factor_score(
        _input(
            gold_price=2280.0,
            ma20=2300.0,
            ma60=2350.0,
            ma120=2325.0,
            macd_value=-9.0,
            macd_signal=-5.0,
            rsi=28.0,
            real_yield_trend="up",
            dxy_trend="flat",
            etf_flow_trend="flat",
            cftc_position_state="neutral",
        )
    )

    assert result.macro_score == -35
    assert result.technical_score == -35
    assert result.flow_score == 0
    assert result.total_score == -26
    assert result.decision == "偏空"
    assert any("价格跌破多条均线" in risk for risk in result.risks)


def test_factor_score_strong_bearish_case() -> None:
    result = calculate_factor_score(
        _input(
            gold_price=2200.0,
            ma20=2250.0,
            ma60=2300.0,
            ma120=2350.0,
            macd_value=-15.0,
            macd_signal=-8.0,
            rsi=82.0,
            real_yield_trend="up",
            dxy_trend="up",
            etf_flow_trend="down",
            cftc_position_state="net_short",
        )
    )

    assert result.macro_score == -65
    assert result.technical_score == -80
    assert result.flow_score == -55
    assert result.total_score == -68
    assert result.decision == "强空"
    assert any("美元继续走强" in risk for risk in result.risks)


def test_factor_score_uses_configurable_parameters() -> None:
    config = FactorScoreConfig(
        technical_weight=1.0,
        macro_weight=0.0,
        flow_weight=0.0,
        macd_bullish_score=40,
        decision_thresholds=DecisionThresholds(
            strong_bullish=80,
            bullish=30,
            bearish=-30,
            strong_bearish=-80,
        ),
    )

    result = calculate_factor_score(
        _input(
            gold_price=2350.0,
            ma20=2360.0,
            ma60=2340.0,
            ma120=2350.0,
            real_yield_trend="flat",
            dxy_trend="flat",
            etf_flow_trend="flat",
            cftc_position_state="neutral",
        ),
        config,
    )

    assert result.technical_score == 40
    assert result.total_score == 40
    assert result.decision == "偏多"


def test_decision_thresholds_validate_ordering() -> None:
    with pytest.raises(ValueError):
        DecisionThresholds(
            strong_bullish=20, bullish=30, bearish=-20, strong_bearish=-60
        )


def test_decide_from_total_score_boundaries() -> None:
    assert decide_from_total_score(60) == "强多"
    assert decide_from_total_score(20) == "偏多"
    assert decide_from_total_score(0) == "中性"
    assert decide_from_total_score(-20) == "偏空"
    assert decide_from_total_score(-60) == "强空"
