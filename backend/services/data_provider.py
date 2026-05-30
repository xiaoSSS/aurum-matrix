"""Unified gold data provider entrypoint.

Business services consume the normalized models from this module instead of raw
FRED or Alpha Vantage client responses. This keeps provider-specific response
shapes isolated at the data-source boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field

from backend.config import AppConfig, get_settings
from data_sources.fred_client import FredClient, FredClientError, FredSeriesData
from data_sources.market_client import (
    AlphaVantageMarketClient,
    DailyBar,
    DailyMarketData,
    MarketClientError,
)
from data_sources.mock_gold import get_mock_gold_price_history, get_mock_gold_snapshot
from scoring.factor_score import CftcPositionState


class UnifiedDailyBar(BaseModel):
    """Provider-neutral daily OHLCV bar."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = Field(default=None, ge=0)


class UnifiedMacroObservation(BaseModel):
    """Provider-neutral macro observation."""

    date: str
    value: float | None = None


class UnifiedGoldData(BaseModel):
    """Provider-neutral data bundle needed by the gold evaluator."""

    mode: str
    provider: str
    as_of: str
    gold_price: float | None = None
    dxy_price: float | None = None
    real_yield_10y: float | None = None
    treasury_yield_10y: float | None = None
    inflation_expectations: float | None = None
    fed_funds_rate: float | None = None
    etf_flow_tonnes_5d: float | None = None
    geopolitical_risk_index: float | None = None
    cftc_position_state: CftcPositionState | None = None
    real_yield_history: list[UnifiedMacroObservation] = Field(default_factory=list)
    treasury_yield_history: list[UnifiedMacroObservation] = Field(default_factory=list)
    inflation_expectations_history: list[UnifiedMacroObservation] = Field(
        default_factory=list
    )
    xau_usd_daily: list[UnifiedDailyBar] = Field(default_factory=list)
    dxy_daily: list[UnifiedDailyBar] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GoldMarketDataClient(Protocol):
    """Protocol for market clients used by the real provider."""

    def get_xau_usd_daily(self, *, outputsize: str = "compact") -> DailyMarketData:
        """Return XAU/USD daily bars."""

    def get_dxy_daily(self, *, outputsize: str = "compact") -> DailyMarketData:
        """Return DXY daily bars."""


class GoldMacroDataClient(Protocol):
    """Protocol for macro clients used by the real provider."""

    def get_10y_real_rate(self) -> FredSeriesData:
        """Return 10-year real-rate observations."""

    def get_10y_treasury_yield(self) -> FredSeriesData:
        """Return 10-year Treasury yield observations."""

    def get_inflation_expectations(self) -> FredSeriesData:
        """Return inflation-expectation observations."""


class GoldDataProvider(Protocol):
    """Unified data-provider interface for gold evaluation."""

    def get_gold_data(self) -> UnifiedGoldData:
        """Return normalized gold evaluation data."""


class MockGoldDataProvider:
    """Deterministic mock provider for tests and local development."""

    def get_gold_data(self) -> UnifiedGoldData:
        """Return normalized mock data."""

        snapshot = get_mock_gold_snapshot()
        history = get_mock_gold_price_history()
        xau_bars = [
            UnifiedDailyBar(
                date=(
                    str(row.date.date()) if hasattr(row.date, "date") else str(row.date)
                ),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=None,
            )
            for row in history.itertuples(index=False)
        ]
        dxy_bars = [
            UnifiedDailyBar(
                date="2026-05-29",
                open=snapshot.dxy + 0.3,
                high=snapshot.dxy + 0.4,
                low=snapshot.dxy + 0.1,
                close=snapshot.dxy + 0.2,
                volume=None,
            ),
            UnifiedDailyBar(
                date=snapshot.as_of.date().isoformat(),
                open=snapshot.dxy + 0.1,
                high=snapshot.dxy + 0.2,
                low=snapshot.dxy - 0.2,
                close=snapshot.dxy,
                volume=None,
            ),
        ]
        return UnifiedGoldData(
            mode="mock",
            provider="mock",
            as_of=snapshot.as_of.isoformat(),
            gold_price=snapshot.price,
            dxy_price=snapshot.dxy,
            real_yield_10y=snapshot.real_yield_10y,
            fed_funds_rate=snapshot.fed_funds_rate,
            etf_flow_tonnes_5d=snapshot.etf_flow_tonnes_5d,
            geopolitical_risk_index=snapshot.geopolitical_risk_index,
            cftc_position_state="net_long_increasing",
            real_yield_history=[
                UnifiedMacroObservation(
                    date="2026-05-29", value=snapshot.real_yield_10y + 0.08
                ),
                UnifiedMacroObservation(
                    date=snapshot.as_of.date().isoformat(),
                    value=snapshot.real_yield_10y,
                ),
            ],
            treasury_yield_history=[
                UnifiedMacroObservation(
                    date="2026-05-29", value=snapshot.fed_funds_rate - 0.20
                ),
                UnifiedMacroObservation(
                    date=snapshot.as_of.date().isoformat(),
                    value=snapshot.fed_funds_rate - 0.25,
                ),
            ],
            inflation_expectations_history=[
                UnifiedMacroObservation(date="2026-05-29", value=2.25),
                UnifiedMacroObservation(
                    date=snapshot.as_of.date().isoformat(), value=2.30
                ),
            ],
            xau_usd_daily=xau_bars,
            dxy_daily=dxy_bars,
        )


class RealGoldDataProvider:
    """Real-data provider backed by Alpha Vantage and FRED clients."""

    def __init__(
        self,
        *,
        settings: AppConfig | None = None,
        market_client: GoldMarketDataClient | None = None,
        macro_client: GoldMacroDataClient | None = None,
    ) -> None:
        """Initialize with explicit clients or construct configured defaults."""

        self.settings = settings or get_settings()
        self.market_client = market_client
        self.macro_client = macro_client

    def get_gold_data(self) -> UnifiedGoldData:
        """Return normalized real market and macro data."""

        errors: list[str] = []
        as_of = datetime.now(timezone.utc).isoformat()
        xau_bars: list[UnifiedDailyBar] = []
        dxy_bars: list[UnifiedDailyBar] = []
        gold_price: float | None = None
        dxy_price: float | None = None
        real_yield_10y: float | None = None
        treasury_yield_10y: float | None = None
        inflation_expectations: float | None = None
        real_yield_history: list[UnifiedMacroObservation] = []
        treasury_yield_history: list[UnifiedMacroObservation] = []
        inflation_expectations_history: list[UnifiedMacroObservation] = []

        try:
            market_client = self.market_client or AlphaVantageMarketClient(
                api_key=self.settings.alpha_vantage_api_key,
                cache_path=self.settings.cache_db_path,
                dxy_symbol=self.settings.default_dxy_symbol,
                settings=self.settings,
            )
            xau_data = market_client.get_xau_usd_daily(outputsize="full")
            xau_bars = _normalize_daily_bars(xau_data.bars)
            gold_price = _latest_close(xau_bars)
            as_of = _latest_date(xau_bars) or as_of
            if gold_price is None:
                errors.append("XAU/USD 日线数据不足，无法读取最新收盘价。")
        except MarketClientError as exc:
            errors.append(f"XAU/USD 行情数据不足：{exc}")

        try:
            market_client = self.market_client or AlphaVantageMarketClient(
                api_key=self.settings.alpha_vantage_api_key,
                cache_path=self.settings.cache_db_path,
                dxy_symbol=self.settings.default_dxy_symbol,
                settings=self.settings,
            )
            dxy_data = market_client.get_dxy_daily(outputsize="compact")
            dxy_bars = _normalize_daily_bars(dxy_data.bars)
            dxy_price = _latest_close(dxy_bars)
            if dxy_price is None:
                errors.append("DXY 日线数据不足，无法读取最新收盘价。")
        except MarketClientError as exc:
            errors.append(f"DXY 行情数据不足：{exc}")

        try:
            macro_client = self.macro_client or FredClient(
                api_key=self.settings.fred_api_key,
                cache_path=self.settings.cache_db_path,
                settings=self.settings,
            )
            real_yield_data = macro_client.get_10y_real_rate()
            real_yield_history = _normalize_observations(real_yield_data)
            real_yield_10y = _latest_observation_value(real_yield_data)
            if real_yield_10y is None:
                errors.append("FRED 10年期实际利率数据不足。")
        except FredClientError as exc:
            errors.append(f"FRED 10年期实际利率数据不足：{exc}")

        try:
            macro_client = self.macro_client or FredClient(
                api_key=self.settings.fred_api_key,
                cache_path=self.settings.cache_db_path,
                settings=self.settings,
            )
            treasury_yield_data = macro_client.get_10y_treasury_yield()
            treasury_yield_history = _normalize_observations(treasury_yield_data)
            treasury_yield_10y = _latest_observation_value(treasury_yield_data)
            if treasury_yield_10y is None:
                errors.append("FRED 10年期美债收益率数据不足。")
        except FredClientError as exc:
            errors.append(f"FRED 10年期美债收益率数据不足：{exc}")

        try:
            macro_client = self.macro_client or FredClient(
                api_key=self.settings.fred_api_key,
                cache_path=self.settings.cache_db_path,
                settings=self.settings,
            )
            inflation_expectations_data = macro_client.get_inflation_expectations()
            inflation_expectations_history = _normalize_observations(
                inflation_expectations_data
            )
            inflation_expectations = _latest_observation_value(
                inflation_expectations_data
            )
            if inflation_expectations is None:
                errors.append("FRED 通胀预期数据不足。")
        except FredClientError as exc:
            errors.append(f"FRED 通胀预期数据不足：{exc}")

        if None in (gold_price, dxy_price, real_yield_10y):
            errors.append("黄金评分所需的核心行情或宏观数据不足。")
        errors.append(
            "ETF 资金流、CFTC 持仓和地缘风险真实数据源尚未接入，暂不计算评分。"
        )

        return UnifiedGoldData(
            mode="real",
            provider="alpha_vantage+fred",
            as_of=as_of,
            gold_price=gold_price,
            dxy_price=dxy_price,
            real_yield_10y=real_yield_10y,
            treasury_yield_10y=treasury_yield_10y,
            inflation_expectations=inflation_expectations,
            real_yield_history=real_yield_history,
            treasury_yield_history=treasury_yield_history,
            inflation_expectations_history=inflation_expectations_history,
            xau_usd_daily=xau_bars,
            dxy_daily=dxy_bars,
            errors=list(dict.fromkeys(errors)),
        )


def create_gold_data_provider(
    *,
    settings: AppConfig | None = None,
    market_client: GoldMarketDataClient | None = None,
    macro_client: GoldMacroDataClient | None = None,
) -> GoldDataProvider:
    """Create the configured mock or real gold data provider."""

    active_settings = settings or get_settings()
    if active_settings.data_mode == "mock":
        return MockGoldDataProvider()
    return RealGoldDataProvider(
        settings=active_settings,
        market_client=market_client,
        macro_client=macro_client,
    )


def _normalize_daily_bars(bars: list[DailyBar]) -> list[UnifiedDailyBar]:
    """Convert client-specific daily bars into provider-neutral bars."""

    return [
        UnifiedDailyBar(
            date=bar.date,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
        for bar in sorted(bars, key=lambda item: item.date)
    ]


def _latest_close(bars: list[UnifiedDailyBar]) -> float | None:
    """Return latest close from normalized daily bars."""

    if not bars:
        return None
    return bars[-1].close


def _latest_date(bars: list[UnifiedDailyBar]) -> str | None:
    """Return latest date from normalized daily bars."""

    if not bars:
        return None
    return bars[-1].date


def _latest_observation_value(data: FredSeriesData) -> float | None:
    """Return latest non-missing FRED observation value."""

    for observation in reversed(data.observations):
        if observation.value is not None:
            return observation.value
    return None


def _normalize_observations(data: FredSeriesData) -> list[UnifiedMacroObservation]:
    """Convert FRED observations into provider-neutral macro observations."""

    return [
        UnifiedMacroObservation(date=observation.date, value=observation.value)
        for observation in sorted(data.observations, key=lambda item: item.date)
    ]
