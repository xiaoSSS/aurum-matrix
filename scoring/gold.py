"""Rule-based gold scoring logic."""

from __future__ import annotations

from data_sources.models import GoldMarketSnapshot
from indicators.technical import TechnicalIndicators
from scoring.models import ScoreBreakdown


def calculate_gold_score(
    snapshot: GoldMarketSnapshot, technical: TechnicalIndicators
) -> ScoreBreakdown:
    """Calculate macro, technical, flow and total scores from concrete inputs."""

    factors: list[str] = []
    macro_score = 0
    if snapshot.real_yield_10y < 1.5:
        macro_score += 25
        factors.append("实际利率偏低，对黄金相对有利")
    elif snapshot.real_yield_10y > 2.2:
        macro_score -= 25
        factors.append("实际利率偏高，压制黄金吸引力")
    else:
        factors.append("实际利率处于中性区间")

    if snapshot.dxy < 102:
        macro_score += 20
        factors.append("美元指数偏弱，利好黄金计价表现")
    elif snapshot.dxy > 106:
        macro_score -= 20
        factors.append("美元指数偏强，对黄金形成压力")
    else:
        factors.append("美元指数处于中性区间")

    if snapshot.geopolitical_risk_index >= 60:
        macro_score += 15
        factors.append("避险情绪较高，提升黄金配置价值")

    technical_score = 0
    if technical.ma_short > technical.ma_long:
        technical_score += 25
        factors.append("短期均线高于长期均线，趋势偏强")
    else:
        technical_score -= 20
        factors.append("短期均线低于长期均线，趋势偏弱")

    if technical.macd > technical.macd_signal:
        technical_score += 15
        factors.append("MACD 高于信号线，动能改善")
    else:
        technical_score -= 15
        factors.append("MACD 低于信号线，动能偏弱")

    if technical.rsi > 70:
        technical_score -= 15
        factors.append("RSI 进入高位区，追高风险上升")
    elif technical.rsi < 30:
        technical_score += 10
        factors.append("RSI 偏低，存在修复可能")
    else:
        factors.append("RSI 未显示极端超买或超卖")

    flow_score = 0
    if snapshot.etf_flow_tonnes_5d > 0:
        flow_score += 20
        factors.append("ETF 资金近五日净流入")
    elif snapshot.etf_flow_tonnes_5d < 0:
        flow_score -= 20
        factors.append("ETF 资金近五日净流出")
    else:
        factors.append("ETF 资金流向中性")

    total_score = round(macro_score * 0.4 + technical_score * 0.4 + flow_score * 0.2)
    if total_score >= 25:
        signal = "bullish"
    elif total_score <= -25:
        signal = "bearish"
    else:
        signal = "neutral"

    return ScoreBreakdown(
        macro_score=macro_score,
        technical_score=technical_score,
        flow_score=flow_score,
        total_score=total_score,
        signal=signal,
        factors=factors,
    )
