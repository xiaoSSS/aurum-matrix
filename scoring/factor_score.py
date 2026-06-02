"""Configurable factor scoring for gold current-price evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from pydantic import BaseModel, Field

from scoring.decision_engine import Decision, DecisionThresholds, decide_from_total_score

Trend = Literal["up", "flat", "down"]
CftcPositionState = Literal[
    "net_long_increasing",
    "neutral",
    "net_short",
    "crowded_long",
]
GeopoliticalRiskState = Literal["high", "neutral", "low"]


class FactorScoreInput(BaseModel):
    """Input factors for rule-based gold scoring."""

    gold_price: float = Field(gt=0)
    ma20: float = Field(gt=0)
    ma60: float = Field(gt=0)
    ma120: float = Field(gt=0)
    macd_value: float
    macd_signal: float
    rsi: float = Field(ge=0, le=100)
    real_yield_trend: Trend
    dxy_trend: Trend
    etf_flow_trend: Trend | None = None
    cftc_position_state: CftcPositionState | None = None
    geopolitical_risk_state: GeopoliticalRiskState | None = None
    valuation_score: int | None = Field(default=None, ge=-100, le=100)


class FactorScoreResult(BaseModel):
    """Structured output for factor scoring."""

    macro_score: int = Field(ge=-100, le=100)
    technical_score: int = Field(ge=-100, le=100)
    flow_score: int = Field(ge=-100, le=100)
    valuation_score: int | None = Field(default=None, ge=-100, le=100)
    total_score: int = Field(ge=-100, le=100)
    decision: Decision
    reasons: list[str]
    risks: list[str]


@dataclass(frozen=True)
class FactorScoreConfig:
    macro_weight: float = 0.35
    technical_weight: float = 0.40
    flow_weight: float = 0.25
    valuation_weight: float = 0.20
    min_score: int = -100
    max_score: int = 100
    decision_thresholds: DecisionThresholds = field(default_factory=DecisionThresholds)

    real_yield_trend_scores: Mapping[Trend, int] = field(
        default_factory=lambda: {"down": 35, "flat": 0, "up": -35}
    )
    dxy_trend_scores: Mapping[Trend, int] = field(
        default_factory=lambda: {"down": 30, "flat": 0, "up": -30}
    )
    etf_flow_trend_scores: Mapping[Trend, int] = field(
        default_factory=lambda: {"up": 35, "flat": 0, "down": -35}
    )
    cftc_position_scores: Mapping[CftcPositionState, int] = field(
        default_factory=lambda: {
            "net_long_increasing": 25,
            "neutral": 0,
            "net_short": -20,
            "crowded_long": -25,
        }
    )
    geopolitical_risk_scores: Mapping[GeopoliticalRiskState, int] = field(
        default_factory=lambda: {"high": 15, "neutral": 0, "low": -10}
    )

    price_above_all_ma_score: int = 25
    price_below_all_ma_score: int = -25
    price_mixed_ma_score: int = 0
    bullish_ma_alignment_score: int = 20
    bearish_ma_alignment_score: int = -20
    mixed_ma_alignment_score: int = 0
    macd_bullish_score: int = 20
    macd_bearish_score: int = -20
    rsi_oversold_score: int = 10
    rsi_overbought_score: int = -15
    rsi_neutral_score: int = 0
    rsi_oversold_threshold: float = 30
    rsi_overbought_threshold: float = 70
    rsi_extreme_low_threshold: float = 20
    rsi_extreme_high_threshold: float = 80


def calculate_factor_score(
    factors: FactorScoreInput, config: FactorScoreConfig | None = None
) -> FactorScoreResult:
    active_config = config or FactorScoreConfig()
    reasons: list[str] = []
    risks: list[str] = []

    macro_score = _score_macro(factors, active_config, reasons, risks)
    technical_score = _score_technical(factors, active_config, reasons, risks)
    flow_score = _score_flow(factors, active_config, reasons, risks)
    total_score = _weighted_total(
        macro_score=macro_score,
        technical_score=technical_score,
        flow_score=flow_score,
        valuation_score=factors.valuation_score,
        config=active_config,
    )
    decision = decide_from_total_score(total_score, active_config.decision_thresholds)

    if not risks:
        risks.append("未发现单一因子极端风险，但本结果不构成收益承诺。")

    return FactorScoreResult(
        macro_score=macro_score,
        technical_score=technical_score,
        flow_score=flow_score,
        valuation_score=factors.valuation_score,
        total_score=total_score,
        decision=decision,
        reasons=reasons,
        risks=risks,
    )


def _score_macro(
    factors: FactorScoreInput,
    config: FactorScoreConfig,
    reasons: list[str],
    risks: list[str],
) -> int:
    real_yield_score = config.real_yield_trend_scores[factors.real_yield_trend]
    dxy_score = config.dxy_trend_scores[factors.dxy_trend]
    if factors.real_yield_trend == "down":
        reasons.append("实际利率趋势下行")
    elif factors.real_yield_trend == "up":
        risks.append("实际利率趋势上行")
    if factors.dxy_trend == "up":
        risks.append("美元继续走强")
    return _clamp(real_yield_score + dxy_score, config)


def _score_technical(
    factors: FactorScoreInput,
    config: FactorScoreConfig,
    reasons: list[str],
    risks: list[str],
) -> int:
    score = 0
    moving_averages = (factors.ma20, factors.ma60, factors.ma120)
    if all(factors.gold_price > average for average in moving_averages):
        score += config.price_above_all_ma_score
    elif all(factors.gold_price < average for average in moving_averages):
        score += config.price_below_all_ma_score
        risks.append("价格跌破多条均线")

    if factors.ma20 > factors.ma60 > factors.ma120:
        score += config.bullish_ma_alignment_score
        reasons.append("均线多头排列")
    elif factors.ma20 < factors.ma60 < factors.ma120:
        score += config.bearish_ma_alignment_score
        reasons.append("均线空头排列")
        risks.append("Bearish moving-average alignment.")

    if factors.macd_value > factors.macd_signal:
        score += config.macd_bullish_score
        reasons.append("MACD 高于信号线")
    else:
        score += config.macd_bearish_score
        reasons.append("MACD 低于信号线")

    if factors.rsi <= config.rsi_oversold_threshold:
        score += config.rsi_oversold_score
    elif factors.rsi >= config.rsi_overbought_threshold:
        score += config.rsi_overbought_score
        reasons.append("RSI 处于高位")

    return _clamp(score, config)


def _score_flow(
    factors: FactorScoreInput,
    config: FactorScoreConfig,
    reasons: list[str],
    risks: list[str],
) -> int:
    score = 0
    if factors.etf_flow_trend is not None:
        score += config.etf_flow_trend_scores[factors.etf_flow_trend]
        if factors.etf_flow_trend == "up":
            reasons.append("黄金 ETF 资金流趋势改善")
        elif factors.etf_flow_trend == "down":
            reasons.append("黄金 ETF 资金流趋势走弱")
            risks.append("ETF 资金持续流出")
        else:
            reasons.append("黄金 ETF 资金流趋势中性")
    if factors.cftc_position_state is not None:
        score += config.cftc_position_scores[factors.cftc_position_state]
        reasons.append("CFTC positioning used")
    if factors.geopolitical_risk_state is not None:
        score += config.geopolitical_risk_scores[factors.geopolitical_risk_state]
        reasons.append("Geopolitical risk used")
    return _clamp(score, config)


def _weighted_total(
    *,
    macro_score: int,
    technical_score: int,
    flow_score: int,
    valuation_score: int | None,
    config: FactorScoreConfig,
) -> int:
    weighted = (
        macro_score * config.macro_weight
        + technical_score * config.technical_weight
        + flow_score * config.flow_weight
    )
    weight_sum = config.macro_weight + config.technical_weight + config.flow_weight
    if valuation_score is not None:
        weighted += valuation_score * config.valuation_weight
        weight_sum += config.valuation_weight
    total = round(weighted / weight_sum)
    return _clamp(total, config)


def _clamp(value: float | int, config: FactorScoreConfig) -> int:
    return int(max(config.min_score, min(config.max_score, round(value))))
