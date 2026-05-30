"""End-to-end gold evaluation service using normalized data providers."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from backend.config import AppConfig, DataMode, get_settings
from backend.services.data_provider import (
    GoldDataProvider,
    GoldMacroDataClient,
    GoldMarketDataClient,
    UnifiedDailyBar,
    UnifiedGoldData,
    UnifiedMacroObservation,
    create_gold_data_provider,
)
from indicators.technical import atr, macd, rsi, simple_moving_average
from llm.report_agent import generate_markdown_report
from scoring.factor_score import (
    CftcPositionState,
    FactorScoreInput,
    FactorScoreResult,
    Trend,
    calculate_factor_score,
)

DataStatus = Literal["ok", "insufficient_data"]
Action = Literal["增配", "持有", "观望", "减配", "回避", "数据不足"]

RISK_DISCLAIMER = (
    "风险提示：本工具仅用于黄金当前价位的规则化评估，不做确定性价格预测，"
    "不提供收益承诺，也不构成投资建议。"
)


class GoldIndicatorValues(BaseModel):
    """Latest technical indicators used by factor scoring."""

    ma20: float = Field(gt=0)
    ma60: float = Field(gt=0)
    ma120: float = Field(gt=0)
    macd_value: float
    macd_signal: float
    rsi: float = Field(ge=0, le=100)
    atr: float = Field(ge=0)


class MacroEnvironment(BaseModel):
    """Macro and DXY inputs normalized for the response."""

    real_yield_10y: float | None = None
    treasury_yield_10y: float | None = None
    inflation_expectations: float | None = None
    real_yield_trend: Trend | None = None
    dxy_trend: Trend | None = None
    etf_flow_trend: Trend | None = None
    cftc_position_state: CftcPositionState | None = None


class EvaluateGoldResponse(BaseModel):
    """Complete API response for the gold evaluation flow."""

    symbol: str = "XAUUSD"
    currency: str = "USD/oz"
    data_status: DataStatus
    data_mode: str
    provider: str
    as_of: str
    price: float | None = None
    dxy_price: float | None = None
    indicators: GoldIndicatorValues | None = None
    macro: MacroEnvironment
    factor_input: FactorScoreInput | None = None
    score: FactorScoreResult | None = None
    market_state: str
    action: Action
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    report_markdown: str
    risk_disclaimer: str = RISK_DISCLAIMER


class GoldEvaluatorService:
    """Application service that owns the gold evaluation business workflow."""

    def __init__(
        self,
        *,
        data_provider: GoldDataProvider | None = None,
        market_client: GoldMarketDataClient | None = None,
        macro_client: GoldMacroDataClient | None = None,
        settings: AppConfig | None = None,
    ) -> None:
        """Initialize the evaluator with optional test/provider dependencies."""

        self.data_provider = data_provider
        self.market_client = market_client
        self.macro_client = macro_client
        self.settings = settings

    def evaluate(self, *, mode: DataMode | None = None) -> EvaluateGoldResponse:
        """Run the full gold evaluation pipeline from data retrieval to report."""

        active_settings = self._settings_for_mode(mode)
        provider = self.data_provider or create_gold_data_provider(
            settings=active_settings,
            market_client=self.market_client,
            macro_client=self.macro_client,
        )
        data = provider.get_gold_data()
        errors = list(data.errors)

        indicators = _calculate_indicator_values(data.xau_usd_daily, errors)
        macro = _build_macro_environment(data, errors)
        factor_input = _build_factor_input(data, indicators, macro, errors)

        score: FactorScoreResult | None = None
        if factor_input is not None:
            score = calculate_factor_score(factor_input)

        market_state = score.decision if score is not None else "数据不足"
        action = _action_from_market_state(market_state)
        reasons = score.reasons if score is not None else []
        risks = score.risks if score is not None else [RISK_DISCLAIMER]
        report_markdown = _generate_report(score, errors)

        return EvaluateGoldResponse(
            data_status="ok" if score is not None else "insufficient_data",
            data_mode=data.mode,
            provider=data.provider,
            as_of=data.as_of,
            price=data.gold_price,
            dxy_price=data.dxy_price,
            indicators=indicators,
            macro=macro,
            factor_input=factor_input,
            score=score,
            market_state=market_state,
            action=action,
            reasons=reasons,
            risks=risks,
            errors=list(dict.fromkeys(errors)),
            report_markdown=report_markdown,
        )

    def _settings_for_mode(self, mode: DataMode | None) -> AppConfig:
        """Return configured settings, optionally overriding data mode."""

        active_settings = self.settings or get_settings()
        if mode is None:
            return active_settings
        return active_settings.model_copy(update={"data_mode": mode})


def evaluate_gold(
    *,
    data_provider: GoldDataProvider | None = None,
    market_client: GoldMarketDataClient | None = None,
    macro_client: GoldMacroDataClient | None = None,
    settings: AppConfig | None = None,
    mode: DataMode | None = None,
) -> EvaluateGoldResponse:
    """Backward-compatible function wrapper around ``GoldEvaluatorService``."""

    return GoldEvaluatorService(
        data_provider=data_provider,
        market_client=market_client,
        macro_client=macro_client,
        settings=settings,
    ).evaluate(mode=mode)


def evaluate_gold_from_data_sources(
    *,
    data_provider: GoldDataProvider | None = None,
    market_client: GoldMarketDataClient | None = None,
    macro_client: GoldMacroDataClient | None = None,
    settings: AppConfig | None = None,
    mode: DataMode | None = None,
) -> EvaluateGoldResponse:
    """Backward-compatible alias for the new backend evaluator."""

    return evaluate_gold(
        data_provider=data_provider,
        market_client=market_client,
        macro_client=macro_client,
        settings=settings,
        mode=mode,
    )


def _calculate_indicator_values(
    bars: list[UnifiedDailyBar], errors: list[str]
) -> GoldIndicatorValues | None:
    """Calculate MA20/MA60/MA120, MACD, RSI and ATR from XAU/USD bars."""

    if not bars:
        errors.append("XAU/USD 日线历史为空，无法计算技术指标。")
        return None

    frame = pd.DataFrame(
        [bar.model_dump() for bar in sorted(bars, key=lambda item: item.date)]
    )
    try:
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        macd_result = macd(close)
        return GoldIndicatorValues(
            ma20=simple_moving_average(close, 20).latest_value,
            ma60=simple_moving_average(close, 60).latest_value,
            ma120=simple_moving_average(close, 120).latest_value,
            macd_value=macd_result.latest_macd,
            macd_signal=macd_result.latest_signal,
            rsi=rsi(close).latest_value,
            atr=atr(high, low, close).latest_value,
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"技术指标数据不足：{exc}")
        return None


def _build_macro_environment(
    data: UnifiedGoldData, errors: list[str]
) -> MacroEnvironment:
    """Derive real-yield, DXY, ETF-flow and CFTC fields for scoring."""

    real_yield_trend = _trend_from_observations(data.real_yield_history)
    if real_yield_trend is None:
        errors.append("FRED 实际利率历史不足，无法计算 real_yield_trend。")

    dxy_trend = _trend_from_bars(data.dxy_daily)
    if dxy_trend is None:
        errors.append("DXY 日线历史不足，无法计算 dxy_trend。")

    etf_flow_trend = _trend_from_etf_flow(data.etf_flow_tonnes_5d)
    if etf_flow_trend is None:
        errors.append("ETF 资金流数据不足，无法计算 etf_flow_trend。")

    if data.cftc_position_state is None:
        errors.append("CFTC 持仓状态数据不足，无法计算 cftc_position_state。")

    return MacroEnvironment(
        real_yield_10y=data.real_yield_10y,
        treasury_yield_10y=data.treasury_yield_10y,
        inflation_expectations=data.inflation_expectations,
        real_yield_trend=real_yield_trend,
        dxy_trend=dxy_trend,
        etf_flow_trend=etf_flow_trend,
        cftc_position_state=data.cftc_position_state,
    )


def _build_factor_input(
    data: UnifiedGoldData,
    indicators: GoldIndicatorValues | None,
    macro: MacroEnvironment,
    errors: list[str],
) -> FactorScoreInput | None:
    """Assemble and validate factor-score input from computed data."""

    missing: list[str] = []
    if data.gold_price is None:
        missing.append("gold_price")
    if indicators is None:
        missing.append("technical_indicators")
    if macro.real_yield_trend is None:
        missing.append("real_yield_trend")
    if macro.dxy_trend is None:
        missing.append("dxy_trend")
    if macro.etf_flow_trend is None:
        missing.append("etf_flow_trend")
    if macro.cftc_position_state is None:
        missing.append("cftc_position_state")

    if missing:
        errors.append(f"factor_score 输入数据不足：{', '.join(missing)}。")
        return None

    assert indicators is not None
    assert data.gold_price is not None
    assert macro.real_yield_trend is not None
    assert macro.dxy_trend is not None
    assert macro.etf_flow_trend is not None
    assert macro.cftc_position_state is not None

    return FactorScoreInput(
        gold_price=data.gold_price,
        ma20=indicators.ma20,
        ma60=indicators.ma60,
        ma120=indicators.ma120,
        macd_value=indicators.macd_value,
        macd_signal=indicators.macd_signal,
        rsi=indicators.rsi,
        real_yield_trend=macro.real_yield_trend,
        dxy_trend=macro.dxy_trend,
        etf_flow_trend=macro.etf_flow_trend,
        cftc_position_state=macro.cftc_position_state,
    )


def _trend_from_observations(
    observations: list[UnifiedMacroObservation], *, tolerance: float = 0.0
) -> Trend | None:
    """Return trend from the latest two non-missing macro observations."""

    values = [
        item.value
        for item in sorted(observations, key=lambda observation: observation.date)
        if item.value is not None
    ]
    return _trend_from_values(values, tolerance=tolerance)


def _trend_from_bars(
    bars: list[UnifiedDailyBar], *, tolerance: float = 0.0
) -> Trend | None:
    """Return trend from the latest two DXY closes."""

    values = [bar.close for bar in sorted(bars, key=lambda x: x.date)]
    return _trend_from_values(values, tolerance=tolerance)


def _trend_from_values(
    values: list[float], *, tolerance: float = 0.0
) -> Trend | None:
    """Classify the latest movement as up, down or flat."""

    if len(values) < 2:
        return None
    change = values[-1] - values[-2]
    if abs(change) <= tolerance:
        return "flat"
    return "up" if change > 0 else "down"


def _trend_from_etf_flow(value: float | None) -> Trend | None:
    """Classify latest ETF flow as up/down/flat without inventing values."""

    if value is None:
        return None
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def _action_from_market_state(market_state: str) -> Action:
    """Map factor-score market state to portfolio/trading action."""

    mapping: dict[str, Action] = {
        "强多": "增配",
        "偏多": "持有",
        "中性": "观望",
        "偏空": "减配",
        "强空": "回避",
    }
    return mapping.get(market_state, "数据不足")


def _generate_report(score: FactorScoreResult | None, errors: list[str]) -> str:
    """Generate deterministic Markdown from structured score output only."""

    if score is not None:
        return generate_markdown_report(score)
    return generate_markdown_report(
        {
            "macro_score": None,
            "technical_score": None,
            "flow_score": None,
            "total_score": None,
            "decision": "",
            "reasons": [],
            "risks": errors or [RISK_DISCLAIMER],
        }
    )
