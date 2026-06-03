from __future__ import annotations

from datetime import date, timedelta

from backend.services.backtest import GoldBacktestService
from backend.config import AppConfig
from backend.services.data_provider import (
    GoldDataProvider,
    UnifiedDailyBar,
    UnifiedGoldData,
    UnifiedMacroObservation,
)


class FakeBacktestProvider(GoldDataProvider):
    def get_gold_data(self) -> UnifiedGoldData:
        bars = _bars(220, 2000.0)
        dxy = _bars(220, 100.0)
        real = [
            UnifiedMacroObservation(
                date=(date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                value=2.0 - i * 0.001,
            )
            for i in range(220)
        ]
        inflation = [
            UnifiedMacroObservation(
                date=(date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                value=300.0 + i * 0.01,
            )
            for i in range(220)
        ]
        debt = [
            UnifiedMacroObservation(
                date=(date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                value=34000.0 + i,
            )
            for i in range(220)
        ]
        cb = [
            UnifiedMacroObservation(
                date=(date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                value=35.0 + (i % 12) * 0.5,
            )
            for i in range(220)
        ]
        return UnifiedGoldData(
            mode="test",
            provider="fake",
            as_of=bars[-1].date,
            gold_price=bars[-1].close,
            dxy_price=dxy[-1].close,
            real_yield_history=real,
            inflation_index_history=inflation,
            us_debt_history=debt,
            central_bank_gold_purchase_history=cb,
            xau_usd_daily=bars,
            dxy_daily=dxy,
        )


def _bars(count: int, start: float) -> list[UnifiedDailyBar]:
    first = date(2026, 1, 1)
    out: list[UnifiedDailyBar] = []
    for i in range(count):
        # deterministic small swings
        drift = i * 0.35
        wave = 2.0 if i % 7 < 3 else -1.5
        close = start + drift + wave
        out.append(
            UnifiedDailyBar(
                date=(first + timedelta(days=i)).isoformat(),
                open=close - 0.4,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1000.0,
            )
        )
    return out


def test_backtest_service_returns_enhanced_structures() -> None:
    result = GoldBacktestService(data_provider=FakeBacktestProvider()).run(
        max_points=80, horizons=[1, 3, 5, 10, 20], fee_bps=2.0, slippage_bps=3.0
    )
    assert result.mode == "test"
    assert result.summary.total_days > 0
    assert len(result.points) <= 80
    assert result.summary.turnover >= 0
    assert result.summary.sharpe_annualized_252 is None or isinstance(
        result.summary.sharpe_annualized_252, float
    )
    assert set(result.decision_buckets.keys()) == {
        "strong_bullish",
        "bullish",
        "neutral",
        "bearish",
        "strong_bearish",
    }
    assert "1d" in result.horizon_metrics.fixed_holding
    assert "20d" in result.horizon_metrics.rolling_rebalance
    assert len(result.equity_curve) == len(result.points)
    assert len(result.score_bins) > 0
    assert len(result.factor_contrib) > 0
    assert "buy_and_hold" in result.benchmarks
    assert "ma_strategy" in result.benchmarks


def test_backtest_includes_valuation_model_comparison_when_inputs_available() -> None:
    settings = AppConfig(
        pricing_model_coefficients_quarterly={
            "intercept": 1800,
            "central_bank_gold_purchase_tonnes": 2,
            "us_total_public_debt": 0.01,
            "dxy_price": -2,
        }
    )
    result = GoldBacktestService(
        data_provider=FakeBacktestProvider(), settings=settings
    ).run(max_points=80)

    assert result.valuation_metrics["available"] is True
    assert len(result.valuation_score_bins) > 0
    assert result.model_comparison["available"] is True
    assert result.model_comparison["with_valuation"] is not None


def test_backtest_experiment_walk_forward() -> None:
    result = GoldBacktestService(data_provider=FakeBacktestProvider()).run_experiment(
        horizons=[1, 3, 5],
        train_size=60,
        test_size=20,
        step_size=20,
        fee_bps=1.0,
        slippage_bps=1.0,
    )
    assert result.horizons == [1, 3, 5]
    assert result.summary.total_days >= 0
    assert len(result.walk_forward) >= 1
    first = result.walk_forward[0]
    assert first.train_start <= first.train_end
    assert first.test_start <= first.test_end
    assert len(result.scenarios) == 1


def test_backtest_experiment_supports_grid() -> None:
    result = GoldBacktestService(data_provider=FakeBacktestProvider()).run_experiment(
        horizons=[1, 3],
        fee_bps=1.0,
        slippage_bps=1.0,
        fee_grid_bps=[0.0, 1.0],
        slippage_grid_bps=[0.0, 2.0],
        train_size=60,
        test_size=20,
        step_size=20,
    )
    assert len(result.scenarios) == 4
