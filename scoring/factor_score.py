"""Configurable factor scoring for gold current-price evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from pydantic import BaseModel, Field

from scoring.decision_engine import (
    Decision,
    DecisionThresholds,
    decide_from_total_score,
)

Trend = Literal["up", "flat", "down"]
CftcPositionState = Literal[
    "net_long_increasing",
    "neutral",
    "net_short",
    "crowded_long",
]


class FactorScoreInput(BaseModel):
    """Input factors for rule-based gold scoring.

    All numeric values must come from a data source or an indicator calculation;
    this module never calls an LLM and never invents inputs.
    """

    gold_price: float = Field(gt=0)
    ma20: float = Field(gt=0)
    ma60: float = Field(gt=0)
    ma120: float = Field(gt=0)
    macd_value: float
    macd_signal: float
    rsi: float = Field(ge=0, le=100)
    real_yield_trend: Trend
    dxy_trend: Trend
    etf_flow_trend: Trend
    cftc_position_state: CftcPositionState


class FactorScoreResult(BaseModel):
    """Structured output for factor scoring."""

    macro_score: int = Field(ge=-100, le=100)
    technical_score: int = Field(ge=-100, le=100)
    flow_score: int = Field(ge=-100, le=100)
    total_score: int = Field(ge=-100, le=100)
    decision: Decision
    reasons: list[str]
    risks: list[str]


@dataclass(frozen=True)
class FactorScoreConfig:
    """Configurable rule parameters for factor scoring."""

    macro_weight: float = 0.35
    technical_weight: float = 0.40
    flow_weight: float = 0.25
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

    def __post_init__(self) -> None:
        """Validate configurable weights and thresholds."""

        total_weight = self.macro_weight + self.technical_weight + self.flow_weight
        if total_weight <= 0:
            raise ValueError("factor weights must sum to a positive value")
        if self.min_score >= self.max_score:
            raise ValueError("min_score must be smaller than max_score")
        if not (0 <= self.rsi_extreme_low_threshold <= self.rsi_oversold_threshold):
            raise ValueError("RSI low thresholds are invalid")
        if not (
            self.rsi_overbought_threshold <= self.rsi_extreme_high_threshold <= 100
        ):
            raise ValueError("RSI high thresholds are invalid")


def calculate_factor_score(
    factors: FactorScoreInput, config: FactorScoreConfig | None = None
) -> FactorScoreResult:
    """Calculate macro, technical, flow and total scores from configured rules."""

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
        config=active_config,
    )
    decision = decide_from_total_score(total_score, active_config.decision_thresholds)

    if not risks:
        risks.append("未发现单一因子极端风险，但本结果不构成收益承诺。")

    return FactorScoreResult(
        macro_score=macro_score,
        technical_score=technical_score,
        flow_score=flow_score,
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
    """Score macro factors from real-yield and DXY trends."""

    real_yield_score = config.real_yield_trend_scores[factors.real_yield_trend]
    dxy_score = config.dxy_trend_scores[factors.dxy_trend]

    if factors.real_yield_trend == "down":
        reasons.append("实际利率趋势下行，对黄金估值相对有利。")
    elif factors.real_yield_trend == "up":
        reasons.append("实际利率趋势上行，可能压制黄金配置吸引力。")
        risks.append("实际利率继续上行可能导致黄金承压。")
    else:
        reasons.append("实际利率趋势中性。")

    if factors.dxy_trend == "down":
        reasons.append("美元指数趋势走弱，利好美元计价黄金。")
    elif factors.dxy_trend == "up":
        reasons.append("美元指数趋势走强，对黄金形成压力。")
        risks.append("美元继续走强可能削弱黄金表现。")
    else:
        reasons.append("美元指数趋势中性。")

    return _clamp(real_yield_score + dxy_score, config)


def _score_technical(
    factors: FactorScoreInput,
    config: FactorScoreConfig,
    reasons: list[str],
    risks: list[str],
) -> int:
    """Score price/MA, MACD and RSI technical factors."""

    score = 0
    moving_averages = (factors.ma20, factors.ma60, factors.ma120)
    if all(factors.gold_price > average for average in moving_averages):
        score += config.price_above_all_ma_score
        reasons.append("金价位于 MA20、MA60、MA120 上方，价格结构偏强。")
    elif all(factors.gold_price < average for average in moving_averages):
        score += config.price_below_all_ma_score
        reasons.append("金价位于 MA20、MA60、MA120 下方，价格结构偏弱。")
        risks.append("价格跌破多条均线，趋势延续下行风险较高。")
    else:
        score += config.price_mixed_ma_score
        reasons.append("金价与主要均线关系混合，价格结构中性。")

    if factors.ma20 > factors.ma60 > factors.ma120:
        score += config.bullish_ma_alignment_score
        reasons.append("MA20 > MA60 > MA120，均线多头排列。")
    elif factors.ma20 < factors.ma60 < factors.ma120:
        score += config.bearish_ma_alignment_score
        reasons.append("MA20 < MA60 < MA120，均线空头排列。")
        risks.append("均线空头排列显示中期趋势偏弱。")
    else:
        score += config.mixed_ma_alignment_score
        reasons.append("均线排列未形成明确趋势。")

    if factors.macd_value > factors.macd_signal:
        score += config.macd_bullish_score
        reasons.append("MACD 高于信号线，动能偏多。")
    else:
        score += config.macd_bearish_score
        reasons.append("MACD 低于或等于信号线，动能偏弱。")
        risks.append("MACD 动能偏弱可能带来回撤风险。")

    if factors.rsi <= config.rsi_oversold_threshold:
        score += config.rsi_oversold_score
        reasons.append("RSI 处于低位，短线存在修复可能。")
        if factors.rsi <= config.rsi_extreme_low_threshold:
            risks.append("RSI 极低可能反映趋势性下跌而非确定性反转。")
    elif factors.rsi >= config.rsi_overbought_threshold:
        score += config.rsi_overbought_score
        reasons.append("RSI 处于高位，追高风险上升。")
        if factors.rsi >= config.rsi_extreme_high_threshold:
            risks.append("RSI 极高，短线波动和回撤风险上升。")
    else:
        score += config.rsi_neutral_score
        reasons.append("RSI 未处于极端区间。")

    return _clamp(score, config)


def _score_flow(
    factors: FactorScoreInput,
    config: FactorScoreConfig,
    reasons: list[str],
    risks: list[str],
) -> int:
    """Score fund-flow factors from ETF flow and CFTC positioning state."""

    etf_score = config.etf_flow_trend_scores[factors.etf_flow_trend]
    cftc_score = config.cftc_position_scores[factors.cftc_position_state]

    if factors.etf_flow_trend == "up":
        reasons.append("黄金 ETF 资金流趋势改善，资金面偏多。")
    elif factors.etf_flow_trend == "down":
        reasons.append("黄金 ETF 资金流趋势走弱，资金面偏空。")
        risks.append("ETF 资金持续流出可能加大价格压力。")
    else:
        reasons.append("黄金 ETF 资金流趋势中性。")

    if factors.cftc_position_state == "net_long_increasing":
        reasons.append("CFTC 净多头增加，投机资金偏多。")
    elif factors.cftc_position_state == "net_short":
        reasons.append("CFTC 持仓偏空，投机资金压力较大。")
        risks.append("期货投机持仓偏空可能压制反弹力度。")
    elif factors.cftc_position_state == "crowded_long":
        reasons.append("CFTC 多头拥挤，继续追多性价比下降。")
        risks.append("多头持仓拥挤，一旦获利了结可能放大波动。")
    else:
        reasons.append("CFTC 持仓状态中性。")

    return _clamp(etf_score + cftc_score, config)


def _weighted_total(
    *,
    macro_score: int,
    technical_score: int,
    flow_score: int,
    config: FactorScoreConfig,
) -> int:
    """Calculate a normalized weighted total score."""

    weight_sum = config.macro_weight + config.technical_weight + config.flow_weight
    total = round(
        (
            macro_score * config.macro_weight
            + technical_score * config.technical_weight
            + flow_score * config.flow_weight
        )
        / weight_sum
    )
    return _clamp(total, config)


def _clamp(value: float | int, config: FactorScoreConfig) -> int:
    """Clamp a score into the configured score range."""

    return int(max(config.min_score, min(config.max_score, round(value))))
