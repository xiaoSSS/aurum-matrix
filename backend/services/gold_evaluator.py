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
from scoring.pricing_model import (
    PricingModelInput,
    PricingModelResult,
    calculate_pricing_model,
    select_pricing_coefficients,
)
from data_sources.sqlite_cache import read_pricing_model_coefficients

DataStatus = Literal["ok", "insufficient_data"]
Action = Literal["增配", "持有", "观望", "减配", "回避", "数据不足"]

RISK_DISCLAIMER = "风险提示：本工具仅用于评估，不构成收益承诺或投资建议。"


class GoldIndicatorValues(BaseModel):
    ma20: float = Field(gt=0)
    ma60: float = Field(gt=0)
    ma120: float = Field(gt=0)
    macd_value: float
    macd_signal: float
    rsi: float = Field(ge=0, le=100)
    atr: float = Field(ge=0)


class MacroEnvironment(BaseModel):
    real_yield_10y: float | None = None
    treasury_yield_10y: float | None = None
    inflation_expectations: float | None = None
    inflation_index: float | None = None
    real_yield_trend: Trend | None = None
    dxy_trend: Trend | None = None
    etf_flow_trend: Trend | None = None
    cftc_position_state: CftcPositionState | None = None
    geopolitical_risk_state: str | None = None
    central_bank_gold_purchase_tonnes: float | None = None
    us_total_public_debt: float | None = None


class EvaluateGoldResponse(BaseModel):
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
    valuation: PricingModelResult | None = None
    enhanced_data_status: dict[str, bool] = Field(default_factory=dict)
    market_state: str
    action: Action
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    report_markdown: str
    risk_disclaimer: str = RISK_DISCLAIMER


class GoldEvaluatorService:
    def __init__(
        self,
        *,
        data_provider: GoldDataProvider | None = None,
        market_client: GoldMarketDataClient | None = None,
        macro_client: GoldMacroDataClient | None = None,
        settings: AppConfig | None = None,
    ) -> None:
        self.data_provider = data_provider
        self.market_client = market_client
        self.macro_client = macro_client
        self.settings = settings

    def evaluate(self, *, mode: DataMode | None = None) -> EvaluateGoldResponse:
        active_settings = self._settings_for_mode(mode)
        provider = self.data_provider or create_gold_data_provider(
            settings=active_settings,
            market_client=self.market_client,
            macro_client=self.macro_client,
        )
        data = provider.get_gold_data()
        errors = list(data.errors)
        warnings = list(data.warnings)

        indicators = _calculate_indicator_values(data.xau_usd_daily, errors)
        macro = _build_macro_environment(data, errors)
        valuation = _build_valuation(data, warnings, active_settings)
        factor_input = _build_factor_input(data, indicators, macro, errors, valuation)

        score: FactorScoreResult | None = None
        if factor_input is not None:
            score = calculate_factor_score(factor_input)

        market_state = score.decision if score is not None else "数据不足"
        action = _action_from_market_state(market_state)
        reasons = score.reasons if score is not None else []
        risks = score.risks if score is not None else [RISK_DISCLAIMER]
        report_markdown = _generate_report(score, errors, valuation)

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
            valuation=valuation,
            enhanced_data_status=_enhanced_data_status(data, valuation),
            market_state=market_state,
            action=action,
            reasons=reasons,
            risks=risks,
            errors=list(dict.fromkeys(errors)),
            warnings=list(dict.fromkeys(warnings)),
            report_markdown=report_markdown,
        )

    def _settings_for_mode(self, mode: DataMode | None) -> AppConfig:
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
    if not bars:
        errors.append("XAU/USD history is empty.")
        return None

    frame = pd.DataFrame([bar.model_dump() for bar in sorted(bars, key=lambda item: item.date)])
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
        errors.append(f"Indicator input invalid: {exc}")
        return None


def _build_macro_environment(data: UnifiedGoldData, errors: list[str]) -> MacroEnvironment:
    real_yield_trend = _trend_from_observations(data.real_yield_history)
    if real_yield_trend is None:
        errors.append("Missing real_yield_trend.")

    dxy_trend = _trend_from_bars(data.dxy_daily)
    if dxy_trend is None:
        errors.append("Missing dxy_trend.")

    etf_flow_trend = _trend_from_etf_flow(data.etf_flow_tonnes_5d)
    if etf_flow_trend is None:
        errors.append("Missing etf_flow_trend (enhanced only).")

    if data.cftc_position_state is None:
        errors.append("Missing cftc_position_state (enhanced only).")

    return MacroEnvironment(
        real_yield_10y=data.real_yield_10y,
        treasury_yield_10y=data.treasury_yield_10y,
        inflation_expectations=data.inflation_expectations,
        inflation_index=data.inflation_index,
        real_yield_trend=real_yield_trend,
        dxy_trend=dxy_trend,
        etf_flow_trend=etf_flow_trend,
        cftc_position_state=data.cftc_position_state,
        geopolitical_risk_state=_geopolitical_risk_state(data.geopolitical_risk_index),
        central_bank_gold_purchase_tonnes=data.central_bank_gold_purchase_tonnes,
        us_total_public_debt=data.us_total_public_debt,
    )


def _build_factor_input(
    data: UnifiedGoldData,
    indicators: GoldIndicatorValues | None,
    macro: MacroEnvironment,
    errors: list[str],
    valuation: PricingModelResult | None = None,
) -> FactorScoreInput | None:
    missing: list[str] = []
    if data.gold_price is None:
        missing.append("gold_price")
    if indicators is None:
        missing.append("technical_indicators")
    if macro.real_yield_trend is None:
        missing.append("real_yield_trend")
    if macro.dxy_trend is None:
        missing.append("dxy_trend")

    if missing:
        errors.append("factor_score 输入数据不足：" + ", ".join(missing))
        return None

    assert indicators is not None
    assert data.gold_price is not None
    assert macro.real_yield_trend is not None
    assert macro.dxy_trend is not None

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
        geopolitical_risk_state=macro.geopolitical_risk_state,
        valuation_score=valuation.valuation_score if valuation is not None else None,
    )


def _build_valuation(
    data: UnifiedGoldData, warnings: list[str], settings: AppConfig
) -> PricingModelResult | None:
    required = (
        data.gold_price,
        data.dxy_price,
        data.central_bank_gold_purchase_tonnes,
        data.us_total_public_debt,
    )
    if any(value is None for value in required):
        warnings.append("Pricing model unavailable: missing one or more inputs")
        return None
    frequency, coefficients = _resolve_pricing_coefficients(settings)
    if not coefficients:
        warnings.append("Pricing model unavailable: coefficients are incomplete")
        return None

    result = calculate_pricing_model(
        PricingModelInput(
            gold_price=float(data.gold_price),
            dxy_price=float(data.dxy_price),
            central_bank_gold_purchase_tonnes=float(
                data.central_bank_gold_purchase_tonnes
            ),
            us_total_public_debt=float(data.us_total_public_debt),
            inflation_index=data.inflation_index,
            inflation_multiplier=_inflation_multiplier(data),
            frequency=frequency,
        ),
        coefficients,
    )
    if result is None:
        warnings.append("Pricing model unavailable: coefficients are incomplete")
    return result


def _resolve_pricing_coefficients(
    settings: AppConfig,
) -> tuple[str, dict[str, float]]:
    monthly = dict(settings.pricing_model_coefficients_monthly)
    quarterly = dict(settings.pricing_model_coefficients_quarterly)
    annual = dict(settings.pricing_model_coefficients_annual)
    if settings.cache_db_path is not None:
        cached = read_pricing_model_coefficients(settings.cache_db_path)
        monthly = monthly or _cached_coefficients(cached, "monthly")
        quarterly = quarterly or _cached_coefficients(cached, "quarterly")
        annual = annual or _cached_coefficients(cached, "annual")
    return select_pricing_coefficients(
        frequency=settings.pricing_model_frequency,
        monthly=monthly,
        quarterly=quarterly,
        annual=annual,
        legacy=settings.pricing_model_coefficients,
    )


def _cached_coefficients(
    cached: dict[str, dict[str, object]], frequency: str
) -> dict[str, float]:
    payload = cached.get(frequency, {})
    raw = payload.get("coefficients") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _inflation_multiplier(data: UnifiedGoldData) -> float:
    history = [
        item.value
        for item in sorted(data.inflation_index_history, key=lambda x: x.date)
        if item.value is not None and item.value > 0
    ]
    if data.inflation_index is not None and data.inflation_index > 0:
        current = float(data.inflation_index)
    elif history:
        current = float(history[-1])
    else:
        return 1.0
    base = float(history[0]) if history else current
    if base <= 0:
        return 1.0
    return current / base


def _enhanced_data_status(
    data: UnifiedGoldData, valuation: PricingModelResult | None
) -> dict[str, bool]:
    return {
        "etf": data.etf_flow_tonnes_5d is not None,
        "cftc": data.cftc_position_state is not None,
        "central_bank_gold": data.central_bank_gold_purchase_tonnes is not None,
        "us_debt": data.us_total_public_debt is not None,
        "valuation": valuation is not None,
    }


def _trend_from_observations(
    observations: list[UnifiedMacroObservation], *, tolerance: float = 0.0
) -> Trend | None:
    values = [
        item.value
        for item in sorted(observations, key=lambda observation: observation.date)
        if item.value is not None
    ]
    return _trend_from_values(values, tolerance=tolerance)


def _trend_from_bars(bars: list[UnifiedDailyBar], *, tolerance: float = 0.0) -> Trend | None:
    values = [bar.close for bar in sorted(bars, key=lambda x: x.date)]
    return _trend_from_values(values, tolerance=tolerance)


def _trend_from_values(values: list[float], *, tolerance: float = 0.0) -> Trend | None:
    if len(values) < 2:
        return None
    change = values[-1] - values[-2]
    if abs(change) <= tolerance:
        return "flat"
    return "up" if change > 0 else "down"


def _trend_from_etf_flow(value: float | None) -> Trend | None:
    if value is None:
        return None
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def _geopolitical_risk_state(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 70:
        return "high"
    if value <= 40:
        return "low"
    return "neutral"


def _action_from_market_state(market_state: str) -> Action:
    mapping: dict[str, Action] = {
        "强多": "增配",
        "偏多": "持有",
        "中性": "观望",
        "偏空": "减配",
        "强空": "回避",
    }
    return mapping.get(market_state, "数据不足")


def _generate_report(
    score: FactorScoreResult | None,
    errors: list[str],
    valuation: PricingModelResult | None = None,
) -> str:
    if score is not None:
        payload = score.model_dump()
        if valuation is not None:
            payload["valuation"] = valuation.model_dump()
        return generate_markdown_report(payload)
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
