from __future__ import annotations

from datetime import date, timedelta

from backend.services.data_provider import (
    GoldDataProvider,
    UnifiedDailyBar,
    UnifiedGoldData,
    UnifiedMacroObservation,
)
from backend.services.gold_evaluator import evaluate_gold


class CompleteFakeProvider(GoldDataProvider):
    def get_gold_data(self) -> UnifiedGoldData:
        return UnifiedGoldData(
            mode="test",
            provider="fake",
            as_of="2026-05-29",
            gold_price=2400.0,
            dxy_price=103.5,
            real_yield_10y=1.75,
            treasury_yield_10y=4.45,
            inflation_expectations=2.30,
            fed_funds_rate=5.25,
            etf_flow_tonnes_5d=4.2,
            geopolitical_risk_index=62.0,
            cftc_position_state="net_long_increasing",
            real_yield_history=[
                UnifiedMacroObservation(date="2026-05-28", value=1.85),
                UnifiedMacroObservation(date="2026-05-29", value=1.75),
            ],
            treasury_yield_history=[
                UnifiedMacroObservation(date="2026-05-28", value=4.50),
                UnifiedMacroObservation(date="2026-05-29", value=4.45),
            ],
            inflation_expectations_history=[
                UnifiedMacroObservation(date="2026-05-28", value=2.25),
                UnifiedMacroObservation(date="2026-05-29", value=2.30),
            ],
            xau_usd_daily=_bars(start=2200.0, count=140),
            dxy_daily=[
                UnifiedDailyBar(
                    date="2026-05-28", open=104.2, high=104.5, low=104.0, close=104.0
                ),
                UnifiedDailyBar(
                    date="2026-05-29", open=103.8, high=104.0, low=103.2, close=103.5
                ),
            ],
        )


class MissingFakeProvider(GoldDataProvider):
    def get_gold_data(self) -> UnifiedGoldData:
        return UnifiedGoldData(
            mode="test",
            provider="fake",
            as_of="2026-05-29",
            gold_price=2400.0,
            xau_usd_daily=_bars(start=2200.0, count=30),
            dxy_daily=[],
        )


def _bars(start: float, count: int) -> list[UnifiedDailyBar]:
    first_day = date(2026, 1, 1)
    return [
        UnifiedDailyBar(
            date=(first_day + timedelta(days=index)).isoformat(),
            open=start + index - 0.5,
            high=start + index + 1.0,
            low=start + index - 1.0,
            close=start + index,
            volume=1000.0,
        )
        for index in range(count)
    ]


def test_evaluate_gold_runs_complete_factor_score_flow() -> None:
    response = evaluate_gold(data_provider=CompleteFakeProvider())

    assert response.symbol == "XAUUSD"
    assert response.data_status == "ok"
    assert response.price == 2400.0
    assert response.indicators is not None
    assert response.indicators.ma20 > 0
    assert response.indicators.ma60 > 0
    assert response.indicators.ma120 > 0
    assert response.macro.real_yield_trend == "down"
    assert response.macro.dxy_trend == "down"
    assert response.factor_input is not None
    assert response.score is not None
    assert response.market_state == response.score.decision
    assert response.action in {"增配", "持有", "观望", "减配", "回避"}
    assert "# 黄金当前价位评估报告" in response.report_markdown
    assert "风险提示" in response.risk_disclaimer


def test_evaluate_gold_returns_insufficient_when_required_data_missing() -> None:
    response = evaluate_gold(data_provider=MissingFakeProvider())

    assert response.data_status == "insufficient_data"
    assert response.market_state == "数据不足"
    assert response.action == "数据不足"
    assert response.score is None
    assert response.factor_input is None
    assert response.indicators is None
    assert any("factor_score 输入数据不足" in error for error in response.errors)
    assert "数据不足" in response.report_markdown
