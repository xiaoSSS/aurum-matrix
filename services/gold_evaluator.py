"""Gold evaluation orchestration using a unified data provider."""

from __future__ import annotations

import pandas as pd

from backend.config import AppConfig
from backend.services.data_provider import (
    GoldDataProvider,
    GoldMacroDataClient,
    GoldMarketDataClient,
    UnifiedDailyBar,
    UnifiedGoldData,
    create_gold_data_provider,
)
from data_sources.models import GoldMarketSnapshot
from decision_engine import decide_gold_action
from indicators.technical import TechnicalIndicators, calculate_technical_indicators
from llm import generate_chinese_report
from scoring.gold import calculate_gold_score
from scoring.models import GoldEvaluation

RISK_DISCLAIMER = "风险提示：本工具仅用于黄金当前价位评估，不提供确定性价格预测或收益承诺，交易需自行承担风险。"


def evaluate_gold_from_data_sources(
    *,
    data_provider: GoldDataProvider | None = None,
    market_client: GoldMarketDataClient | None = None,
    macro_client: GoldMacroDataClient | None = None,
    settings: AppConfig | None = None,
) -> GoldEvaluation:
    """Evaluate gold from a normalized data-provider output.

    Business logic consumes only ``UnifiedGoldData`` rather than raw FRED or
    Alpha Vantage responses. If required real-data fields are missing, the
    service returns ``数据不足`` instead of fabricating values.
    """

    provider = data_provider or create_gold_data_provider(
        settings=settings,
        market_client=market_client,
        macro_client=macro_client,
    )
    data = provider.get_gold_data()
    technical = _technical_from_bars(data.xau_usd_daily)
    errors = list(data.errors)
    if technical is None:
        errors.append("XAU/USD 日线数据不足，无法计算完整技术指标。")

    snapshot = _snapshot_from_unified_data(data)
    if snapshot is None:
        return _insufficient_evaluation(
            data=data,
            technical=technical,
            errors=errors,
        )

    score = calculate_gold_score(snapshot, technical) if technical is not None else None
    if score is None:
        return _insufficient_evaluation(
            data=data,
            technical=technical,
            errors=errors,
        )

    decision = decide_gold_action(score)
    evaluation = GoldEvaluation(
        symbol=snapshot.symbol,
        price=snapshot.price,
        currency=snapshot.currency,
        as_of=data.as_of,
        data_status="ok",
        decision=decision,
        score=score,
        market_data=snapshot,
        technical_indicators=technical,
        summary="",
        risk_disclaimer=RISK_DISCLAIMER,
    )
    evaluation.summary = generate_chinese_report(evaluation)
    return evaluation


def _snapshot_from_unified_data(data: UnifiedGoldData) -> GoldMarketSnapshot | None:
    """Build a scoring snapshot only when all required fields are present."""

    required_values = (
        data.gold_price,
        data.dxy_price,
        data.real_yield_10y,
        data.fed_funds_rate,
        data.etf_flow_tonnes_5d,
        data.geopolitical_risk_index,
    )
    if any(value is None for value in required_values) or data.errors:
        return None
    return GoldMarketSnapshot(
        price=data.gold_price,
        dxy=data.dxy_price,
        real_yield_10y=data.real_yield_10y,
        fed_funds_rate=data.fed_funds_rate,
        etf_flow_tonnes_5d=data.etf_flow_tonnes_5d,
        geopolitical_risk_index=data.geopolitical_risk_index,
        source=data.provider,
    )


def _technical_from_bars(bars: list[UnifiedDailyBar]) -> TechnicalIndicators | None:
    """Convert normalized bars to a DataFrame and calculate indicators."""

    frame = pd.DataFrame([bar.model_dump() for bar in bars])
    if frame.empty:
        return None
    frame = frame.sort_values("date").reset_index(drop=True)
    return calculate_technical_indicators(frame)


def _insufficient_evaluation(
    *,
    data: UnifiedGoldData,
    technical: TechnicalIndicators | None,
    errors: list[str],
) -> GoldEvaluation:
    """Build a structured insufficient-data response."""

    unique_errors = list(dict.fromkeys(errors))
    summary = "数据不足，无法完成黄金当前价位评估。"
    if unique_errors:
        summary = f"{summary} 原因：" + "；".join(unique_errors)
    return GoldEvaluation(
        symbol="XAUUSD",
        price=data.gold_price,
        currency="USD/oz",
        as_of=data.as_of,
        data_status="insufficient_data",
        decision="数据不足",
        score=None,
        market_data=None,
        technical_indicators=technical,
        summary=summary,
        risk_disclaimer=RISK_DISCLAIMER,
    )
