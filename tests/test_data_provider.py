from __future__ import annotations

from backend.config import AppConfig
from backend.services.data_provider import (
    MockGoldDataProvider,
    RealGoldDataProvider,
    create_gold_data_provider,
)
from data_sources.fred_client import FredObservation, FredSeriesData
from data_sources.market_client import DailyBar, DailyMarketData


class FakeMarketClient:
    def get_xau_usd_daily(self, *, outputsize: str = "compact") -> DailyMarketData:
        return DailyMarketData(
            symbol="XAU/USD",
            provider="fake",
            bars=[
                DailyBar(date="2026-05-28", open=1, high=2, low=1, close=1.5),
                DailyBar(date="2026-05-29", open=2, high=3, low=2, close=2.5),
            ],
        )

    def get_dxy_daily(self, *, outputsize: str = "compact") -> DailyMarketData:
        return DailyMarketData(
            symbol="DXY",
            provider="fake",
            bars=[
                DailyBar(date="2026-05-29", open=104, high=105, low=103, close=104.2)
            ],
        )


class FakeMacroClient:
    def get_10y_real_rate(self) -> FredSeriesData:
        return _series("DFII10", 1.84)

    def get_10y_treasury_yield(self) -> FredSeriesData:
        return _series("DGS10", 4.5)

    def get_inflation_expectations(self) -> FredSeriesData:
        return _series("T10YIE", 2.3)


def _series(series_id: str, value: float) -> FredSeriesData:
    return FredSeriesData(
        series_id=series_id,
        observations=[FredObservation(date="2026-05-29", value=value)],
    )


def test_create_gold_data_provider_returns_mock_provider() -> None:
    provider = create_gold_data_provider(settings=AppConfig(data_mode="mock"))

    assert isinstance(provider, MockGoldDataProvider)
    data = provider.get_gold_data()
    assert data.mode == "mock"
    assert data.gold_price == 2368.4
    assert data.dxy_price == 104.2
    assert data.real_yield_10y == 1.84
    assert data.etf_flow_tonnes_5d == 3.7
    assert data.cftc_position_state == "net_long_increasing"
    assert data.real_yield_history[-1].value == 1.84
    assert len(data.dxy_daily) == 2
    assert data.xau_usd_daily


def test_real_gold_data_provider_outputs_unified_structure() -> None:
    provider = RealGoldDataProvider(
        settings=AppConfig(),
        market_client=FakeMarketClient(),
        macro_client=FakeMacroClient(),
    )

    data = provider.get_gold_data()

    assert data.mode == "real"
    assert data.provider == "alpha_vantage+fred"
    assert data.gold_price == 2.5
    assert data.dxy_price == 104.2
    assert data.real_yield_10y == 1.84
    assert data.treasury_yield_10y == 4.5
    assert data.inflation_expectations == 2.3
    assert data.real_yield_history[-1].value == 1.84
    assert data.treasury_yield_history[-1].value == 4.5
    assert data.inflation_expectations_history[-1].value == 2.3
    assert data.xau_usd_daily[-1].date == "2026-05-29"
    assert any("ETF flow data unavailable" in warning for warning in data.warnings)
