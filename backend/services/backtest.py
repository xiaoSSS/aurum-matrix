"""Enhanced historical backtesting for the current gold scoring model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import pstdev
from typing import Any, Literal

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
from indicators.technical import macd, rsi, simple_moving_average
from scoring.factor_score import FactorScoreInput, FactorScoreResult, calculate_factor_score
from scoring.pricing_model import PricingModelInput, calculate_pricing_model, select_pricing_coefficients
from data_sources.sqlite_cache import read_pricing_model_coefficients

DecisionBucketName = Literal[
    "strong_bullish",
    "bullish",
    "neutral",
    "bearish",
    "strong_bearish",
]


class BacktestPoint(BaseModel):
    date: str
    close: float
    total_score: int
    decision: str
    position: float
    gross_daily_return: float
    net_daily_return: float
    turnover: float
    cost_return: float
    forward_returns: dict[str, float | None] = Field(default_factory=dict)
    factor_input: dict[str, Any] = Field(default_factory=dict)


class DecisionBucketMetrics(BaseModel):
    count: int
    accuracy: float | None = None
    avg_return: float | None = None
    strategy_return: float | None = None


class HorizonMetric(BaseModel):
    avg_return: float | None = None
    strategy_return: float | None = None
    accuracy: float | None = None


class HorizonMetricsPair(BaseModel):
    fixed_holding: dict[str, HorizonMetric] = Field(default_factory=dict)
    rolling_rebalance: dict[str, HorizonMetric] = Field(default_factory=dict)


class EquityPoint(BaseModel):
    date: str
    equity: float


class ScoreBinMetric(BaseModel):
    bin_label: str
    lower: float
    upper: float
    count: int
    avg_forward_return: float | None = None
    hit_rate: float | None = None


class FactorContributionMetric(BaseModel):
    trigger: str
    count: int
    avg_forward_return_1d: float | None = None
    hit_rate_1d: float | None = None


class BenchmarkMetric(BaseModel):
    cumulative_return: float
    max_drawdown: float
    sharpe_annualized_252: float | None = None
    gross_strategy_return: float = 0.0
    net_strategy_return: float = 0.0


class BacktestSummary(BaseModel):
    total_days: int
    scored_days: int
    bullish_signals: int
    bearish_signals: int
    neutral_signals: int
    directional_accuracy: float | None = None
    avg_next_day_return: float | None = None
    strategy_avg_return: float | None = None
    cumulative_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_annualized_252: float | None = None
    turnover: float = 0.0
    gross_strategy_return: float = 0.0
    net_strategy_return: float = 0.0


class WalkForwardWindowResult(BaseModel):
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    net_strategy_return: float
    sharpe_annualized_252: float | None = None
    directional_accuracy: float | None = None


class BacktestResult(BaseModel):
    as_of: str | None = None
    mode: str
    provider: str
    summary: BacktestSummary
    points: list[BacktestPoint] = Field(default_factory=list)
    decision_buckets: dict[DecisionBucketName, DecisionBucketMetrics] = Field(default_factory=dict)
    horizon_metrics: HorizonMetricsPair = Field(default_factory=HorizonMetricsPair)
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    score_bins: list[ScoreBinMetric] = Field(default_factory=list)
    factor_contrib: list[FactorContributionMetric] = Field(default_factory=list)
    benchmarks: dict[str, BenchmarkMetric] = Field(default_factory=dict)
    valuation_metrics: dict[str, Any] = Field(default_factory=dict)
    valuation_score_bins: list[ScoreBinMetric] = Field(default_factory=list)
    model_comparison: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class BacktestExperimentResult(BaseModel):
    horizons: list[int]
    fee_bps: float
    slippage_bps: float
    walk_forward: list[WalkForwardWindowResult] = Field(default_factory=list)
    scenarios: list[ExperimentScenarioResult] = Field(default_factory=list)
    summary: BacktestSummary
    errors: list[str] = Field(default_factory=list)


class ExperimentScenarioResult(BaseModel):
    fee_bps: float
    slippage_bps: float
    summary: BacktestSummary
    walk_forward: list[WalkForwardWindowResult] = Field(default_factory=list)


@dataclass
class _RawPoint:
    date: str
    close: float
    score: FactorScoreResult
    factors: FactorScoreInput
    direction_1d: int
    ret_1d: float
    forward_returns: dict[int, float | None]


class GoldBacktestService:
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

    def run(
        self,
        *,
        mode: DataMode | None = None,
        max_points: int = 250,
        horizons: list[int] | None = None,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        position_scheme: str = "strong1_weak0.5",
        score_bins_count: int = 10,
    ) -> BacktestResult:
        horizons = horizons or [1, 3, 5, 10, 20]
        data = self._load_data(mode)
        active_settings = self.settings or get_settings()
        if mode is not None:
            active_settings = active_settings.model_copy(update={"data_mode": mode})
        raw_points, extra_errors = _build_raw_points(
            data.xau_usd_daily,
            data.dxy_daily,
            data.real_yield_history,
            horizons=horizons,
        )
        if max_points > 0:
            raw_points = raw_points[-max_points:]

        strategy = _compute_strategy_returns(
            raw_points, fee_bps=fee_bps, slippage_bps=slippage_bps, position_scheme=position_scheme
        )
        points = _to_points(raw_points, strategy)
        summary = _summarize(points)
        decision_buckets = _decision_buckets(raw_points, position_scheme=position_scheme)
        horizon_metrics = _horizon_metrics(raw_points, horizons=horizons, position_scheme=position_scheme)
        equity_curve = _equity_curve(points)
        score_bins = _score_bin_metrics(raw_points, bins_count=score_bins_count)
        factor_contrib = _factor_contribution(raw_points)
        benchmarks = _benchmarks(raw_points, fee_bps=fee_bps, slippage_bps=slippage_bps)
        valuation = _valuation_backtest(
            raw_points=raw_points,
            data=data,
            settings=active_settings,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            position_scheme=position_scheme,
            score_bins_count=score_bins_count,
        )

        return BacktestResult(
            as_of=data.as_of,
            mode=data.mode,
            provider=data.provider,
            summary=summary,
            points=points,
            decision_buckets=decision_buckets,
            horizon_metrics=horizon_metrics,
            equity_curve=equity_curve,
            score_bins=score_bins,
            factor_contrib=factor_contrib,
            benchmarks=benchmarks,
            valuation_metrics=valuation["valuation_metrics"],
            valuation_score_bins=valuation["valuation_score_bins"],
            model_comparison=valuation["model_comparison"] | {"base_model": summary.model_dump()},
            errors=list(dict.fromkeys([*data.errors, *extra_errors])),
        )

    def run_experiment(
        self,
        *,
        mode: DataMode | None = None,
        horizons: list[int] | None = None,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        fee_grid_bps: list[float] | None = None,
        slippage_grid_bps: list[float] | None = None,
        train_size: int = 120,
        test_size: int = 20,
        step_size: int = 20,
        position_scheme: str = "strong1_weak0.5",
    ) -> BacktestExperimentResult:
        horizons = horizons or [1, 3, 5, 10, 20]
        data = self._load_data(mode)
        raw_points, extra_errors = _build_raw_points(
            data.xau_usd_daily,
            data.dxy_daily,
            data.real_yield_history,
            horizons=horizons,
        )
        scenarios: list[ExperimentScenarioResult] = []
        fee_candidates = fee_grid_bps or [fee_bps]
        slippage_candidates = slippage_grid_bps or [slippage_bps]
        default_windows: list[WalkForwardWindowResult] = []

        for fee_candidate in fee_candidates:
            for slippage_candidate in slippage_candidates:
                windows = _walk_forward_windows(
                    raw_points=raw_points,
                    train_size=train_size,
                    test_size=test_size,
                    step_size=step_size,
                    fee_bps=fee_candidate,
                    slippage_bps=slippage_candidate,
                    position_scheme=position_scheme,
                )
                if fee_candidate == fee_bps and slippage_candidate == slippage_bps:
                    default_windows = windows
                full = self.run(
                    mode=mode,
                    max_points=250,
                    horizons=horizons,
                    fee_bps=fee_candidate,
                    slippage_bps=slippage_candidate,
                    position_scheme=position_scheme,
                )
                scenarios.append(
                    ExperimentScenarioResult(
                        fee_bps=fee_candidate,
                        slippage_bps=slippage_candidate,
                        summary=full.summary,
                        walk_forward=windows,
                    )
                )

        overall = self.run(
            mode=mode,
            max_points=250,
            horizons=horizons,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            position_scheme=position_scheme,
        )
        return BacktestExperimentResult(
            horizons=horizons,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            walk_forward=default_windows,
            scenarios=scenarios,
            summary=overall.summary,
            errors=list(dict.fromkeys([*overall.errors, *extra_errors])),
        )

    def _load_data(self, mode: DataMode | None):
        active_settings = self.settings or get_settings()
        if mode is not None:
            active_settings = active_settings.model_copy(update={"data_mode": mode})
        provider = self.data_provider or create_gold_data_provider(
            settings=active_settings,
            market_client=self.market_client,
            macro_client=self.macro_client,
        )
        return provider.get_gold_data()


def _build_raw_points(
    xau_bars: list[UnifiedDailyBar],
    dxy_bars: list[UnifiedDailyBar],
    real_yield_history: list[UnifiedMacroObservation],
    *,
    horizons: list[int],
) -> tuple[list[_RawPoint], list[str]]:
    errors: list[str] = []
    lookback = 120
    if len(xau_bars) < lookback + 2:
        return [], ["Insufficient XAU history for backtest (need at least 122 bars)."]

    xau = sorted(xau_bars, key=lambda b: b.date)
    dxy = sorted(dxy_bars, key=lambda b: b.date)
    real = sorted(real_yield_history, key=lambda x: x.date)
    points: list[_RawPoint] = []
    max_h = max(horizons) if horizons else 1
    for i in range(lookback, len(xau) - 1):
        today = xau[i]
        closes = pd.Series([bar.close for bar in xau[: i + 1]])
        try:
            ma20 = simple_moving_average(closes, 20).latest_value
            ma60 = simple_moving_average(closes, 60).latest_value
            ma120 = simple_moving_average(closes, 120).latest_value
            macd_res = macd(closes)
            rsi_val = rsi(closes).latest_value
        except Exception:
            continue

        dxy_trend = _trend_from_bars_until(dxy, today.date)
        real_trend = _trend_from_obs_until(real, today.date)
        if dxy_trend is None or real_trend is None:
            continue

        factors = FactorScoreInput(
            gold_price=today.close,
            ma20=ma20,
            ma60=ma60,
            ma120=ma120,
            macd_value=macd_res.latest_macd,
            macd_signal=macd_res.latest_signal,
            rsi=rsi_val,
            real_yield_trend=real_trend,
            dxy_trend=dxy_trend,
        )
        score = calculate_factor_score(factors)
        next_close = xau[i + 1].close
        ret_1d = (next_close - today.close) / today.close
        forward: dict[int, float | None] = {}
        for h in horizons:
            if i + h < len(xau):
                future_close = xau[i + h].close
                forward[h] = (future_close - today.close) / today.close
            else:
                forward[h] = None
        if i + max_h >= len(xau):
            # still keep points if 1d available; higher horizon will be None
            pass
        points.append(
            _RawPoint(
                date=today.date,
                close=today.close,
                score=score,
                factors=factors,
                direction_1d=1 if ret_1d > 0 else (-1 if ret_1d < 0 else 0),
                ret_1d=ret_1d,
                forward_returns=forward,
            )
        )
    return points, errors


def _walk_forward_windows(
    *,
    raw_points: list[_RawPoint],
    train_size: int,
    test_size: int,
    step_size: int,
    fee_bps: float,
    slippage_bps: float,
    position_scheme: str,
) -> list[WalkForwardWindowResult]:
    windows: list[WalkForwardWindowResult] = []
    if len(raw_points) < (train_size + test_size):
        return windows
    start = 0
    while (start + train_size + test_size) <= len(raw_points):
        train = raw_points[start : start + train_size]
        test = raw_points[start + train_size : start + train_size + test_size]
        strategy = _compute_strategy_returns(
            test, fee_bps=fee_bps, slippage_bps=slippage_bps, position_scheme=position_scheme
        )
        points = _to_points(test, strategy)
        summary = _summarize(points)
        windows.append(
            WalkForwardWindowResult(
                train_start=train[0].date,
                train_end=train[-1].date,
                test_start=test[0].date,
                test_end=test[-1].date,
                net_strategy_return=summary.net_strategy_return,
                sharpe_annualized_252=summary.sharpe_annualized_252,
                directional_accuracy=summary.directional_accuracy,
            )
        )
        start += step_size
    return windows


def _decision_alias(decision: str) -> DecisionBucketName:
    if decision in {"强多", "寮哄"}:
        return "strong_bullish"
    if decision in {"偏多", "鍋忓"}:
        return "bullish"
    if decision in {"偏空", "鍋忕┖"}:
        return "bearish"
    if decision in {"强空", "寮虹┖"}:
        return "strong_bearish"
    return "neutral"


def _position_for_decision(decision: str, *, scheme: str) -> float:
    bucket = _decision_alias(decision)
    if scheme != "strong1_weak0.5":
        scheme = "strong1_weak0.5"
    mapping = {
        "strong_bullish": 1.0,
        "bullish": 0.5,
        "neutral": 0.0,
        "bearish": -0.5,
        "strong_bearish": -1.0,
    }
    return mapping[bucket]


def _compute_strategy_returns(
    raw_points: list[_RawPoint],
    *,
    fee_bps: float,
    slippage_bps: float,
    position_scheme: str,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    prev_pos = 0.0
    one_bps = 0.0001
    cost_rate = (fee_bps + slippage_bps) * one_bps
    for p in raw_points:
        pos = _position_for_decision(p.score.decision, scheme=position_scheme)
        turnover = abs(pos - prev_pos)
        cost = turnover * cost_rate
        gross = pos * p.ret_1d
        net = gross - cost
        out.append(
            {
                "position": pos,
                "turnover": turnover,
                "cost": cost,
                "gross": gross,
                "net": net,
            }
        )
        prev_pos = pos
    return out


def _to_points(raw_points: list[_RawPoint], strategy: list[dict[str, float]]) -> list[BacktestPoint]:
    points: list[BacktestPoint] = []
    for rp, st in zip(raw_points, strategy):
        forward = {f"{h}d": v for h, v in rp.forward_returns.items()}
        points.append(
            BacktestPoint(
                date=rp.date,
                close=rp.close,
                total_score=rp.score.total_score,
                decision=rp.score.decision,
                position=st["position"],
                gross_daily_return=st["gross"],
                net_daily_return=st["net"],
                turnover=st["turnover"],
                cost_return=st["cost"],
                forward_returns=forward,
                factor_input=rp.factors.model_dump(),
            )
        )
    return points


def _summarize(points: list[BacktestPoint]) -> BacktestSummary:
    if not points:
        return BacktestSummary(
            total_days=0,
            scored_days=0,
            bullish_signals=0,
            bearish_signals=0,
            neutral_signals=0,
        )
    bullish = sum(1 for p in points if p.position > 0)
    bearish = sum(1 for p in points if p.position < 0)
    neutral = sum(1 for p in points if p.position == 0)
    directional_hits = [
        ((1 if p.position > 0 else -1) == (1 if p.forward_returns.get("1d", 0) > 0 else -1))
        for p in points
        if p.position != 0 and p.forward_returns.get("1d") is not None and p.forward_returns.get("1d") != 0
    ]
    acc = (sum(1 for x in directional_hits if x) / len(directional_hits)) if directional_hits else None
    avg_ret = sum(float(p.forward_returns.get("1d") or 0.0) for p in points) / len(points)
    strat_avg = sum(p.net_daily_return for p in points) / len(points)
    cumulative = _cumulative_return([p.net_daily_return for p in points])
    drawdown = _max_drawdown([p.net_daily_return for p in points])
    sharpe = _sharpe_annualized_252([p.net_daily_return for p in points])
    turnover = sum(p.turnover for p in points)
    gross = sum(p.gross_daily_return for p in points)
    net = sum(p.net_daily_return for p in points)
    return BacktestSummary(
        total_days=len(points),
        scored_days=len(points),
        bullish_signals=bullish,
        bearish_signals=bearish,
        neutral_signals=neutral,
        directional_accuracy=acc,
        avg_next_day_return=avg_ret,
        strategy_avg_return=strat_avg,
        cumulative_return=cumulative,
        max_drawdown=drawdown,
        sharpe_annualized_252=sharpe,
        turnover=turnover,
        gross_strategy_return=gross,
        net_strategy_return=net,
    )


def _decision_buckets(raw_points: list[_RawPoint], *, position_scheme: str) -> dict[DecisionBucketName, DecisionBucketMetrics]:
    grouped: dict[DecisionBucketName, list[_RawPoint]] = {
        "strong_bullish": [],
        "bullish": [],
        "neutral": [],
        "bearish": [],
        "strong_bearish": [],
    }
    for p in raw_points:
        grouped[_decision_alias(p.score.decision)].append(p)
    result: dict[DecisionBucketName, DecisionBucketMetrics] = {}
    for name, items in grouped.items():
        if not items:
            result[name] = DecisionBucketMetrics(count=0)
            continue
        pos = _position_for_decision(items[0].score.decision, scheme=position_scheme)
        rets = [x.ret_1d for x in items]
        non_neutral = [x for x in items if pos != 0]
        hits = [(_sign(x.ret_1d) == _sign(pos)) for x in non_neutral if _sign(x.ret_1d) != 0]
        acc = (sum(1 for x in hits if x) / len(hits)) if hits else None
        result[name] = DecisionBucketMetrics(
            count=len(items),
            accuracy=acc,
            avg_return=sum(rets) / len(rets),
            strategy_return=sum(r * pos for r in rets) / len(rets),
        )
    return result


def _horizon_metrics(raw_points: list[_RawPoint], *, horizons: list[int], position_scheme: str) -> HorizonMetricsPair:
    fixed: dict[str, HorizonMetric] = {}
    rolling: dict[str, HorizonMetric] = {}
    for h in horizons:
        key = f"{h}d"
        vals = [(p.forward_returns.get(h), _position_for_decision(p.score.decision, scheme=position_scheme)) for p in raw_points]
        vals = [(r, pos) for r, pos in vals if r is not None]
        if not vals:
            fixed[key] = HorizonMetric()
            rolling[key] = HorizonMetric()
            continue
        avg = sum(float(r) for r, _ in vals) / len(vals)
        strat = sum(float(r) * pos for r, pos in vals) / len(vals)
        hits = [
            (_sign(float(r)) == _sign(pos))
            for r, pos in vals
            if pos != 0 and _sign(float(r)) != 0
        ]
        acc = (sum(1 for h_ in hits if h_) / len(hits)) if hits else None
        metric = HorizonMetric(avg_return=avg, strategy_return=strat, accuracy=acc)
        fixed[key] = metric
        # rolling horizon: mean of daily strategy returns over rolling windows
        net_series = [float(r) * pos for r, pos in vals]
        rolling_metric = HorizonMetric(
            avg_return=avg,
            strategy_return=_rolling_window_mean(net_series, h),
            accuracy=acc,
        )
        rolling[key] = rolling_metric
    return HorizonMetricsPair(fixed_holding=fixed, rolling_rebalance=rolling)


def _equity_curve(points: list[BacktestPoint]) -> list[EquityPoint]:
    eq = 1.0
    out: list[EquityPoint] = []
    for p in points:
        eq *= (1.0 + p.net_daily_return)
        out.append(EquityPoint(date=p.date, equity=eq))
    return out


def _score_bin_metrics(raw_points: list[_RawPoint], *, bins_count: int) -> list[ScoreBinMetric]:
    if not raw_points:
        return []
    scores = [p.score.total_score for p in raw_points]
    lo, hi = min(scores), max(scores)
    if lo == hi:
        rets = [p.ret_1d for p in raw_points]
        return [
            ScoreBinMetric(
                bin_label=f"[{lo},{hi}]",
                lower=float(lo),
                upper=float(hi),
                count=len(raw_points),
                avg_forward_return=sum(rets) / len(rets),
                hit_rate=sum(1 for x in rets if x > 0) / len(rets),
            )
        ]
    width = (hi - lo) / bins_count
    out: list[ScoreBinMetric] = []
    for i in range(bins_count):
        lower = lo + i * width
        upper = hi if i == bins_count - 1 else (lo + (i + 1) * width)
        items = [p for p in raw_points if (p.score.total_score >= lower and (p.score.total_score <= upper if i == bins_count - 1 else p.score.total_score < upper))]
        if not items:
            out.append(ScoreBinMetric(bin_label=f"[{lower:.1f},{upper:.1f})", lower=lower, upper=upper, count=0))
            continue
        rets = [p.ret_1d for p in items]
        out.append(
            ScoreBinMetric(
                bin_label=f"[{lower:.1f},{upper:.1f}{']' if i == bins_count - 1 else ')'}",
                lower=lower,
                upper=upper,
                count=len(items),
                avg_forward_return=sum(rets) / len(rets),
                hit_rate=sum(1 for r in rets if r > 0) / len(rets),
            )
        )
    return out


def _factor_contribution(raw_points: list[_RawPoint]) -> list[FactorContributionMetric]:
    groups: dict[str, list[_RawPoint]] = {
        "real_yield_down": [p for p in raw_points if p.factors.real_yield_trend == "down"],
        "real_yield_up": [p for p in raw_points if p.factors.real_yield_trend == "up"],
        "dxy_down": [p for p in raw_points if p.factors.dxy_trend == "down"],
        "dxy_up": [p for p in raw_points if p.factors.dxy_trend == "up"],
        "ma_bull": [p for p in raw_points if p.factors.ma20 > p.factors.ma60 > p.factors.ma120],
        "ma_bear": [p for p in raw_points if p.factors.ma20 < p.factors.ma60 < p.factors.ma120],
        "macd_bull": [p for p in raw_points if p.factors.macd_value > p.factors.macd_signal],
        "macd_bear": [p for p in raw_points if p.factors.macd_value <= p.factors.macd_signal],
        "rsi_overbought": [p for p in raw_points if p.factors.rsi >= 70],
        "rsi_oversold": [p for p in raw_points if p.factors.rsi <= 30],
    }
    out: list[FactorContributionMetric] = []
    for name, items in groups.items():
        if not items:
            out.append(FactorContributionMetric(trigger=name, count=0))
            continue
        rets = [p.ret_1d for p in items]
        out.append(
            FactorContributionMetric(
                trigger=name,
                count=len(items),
                avg_forward_return_1d=sum(rets) / len(rets),
                hit_rate_1d=sum(1 for r in rets if r > 0) / len(rets),
            )
        )
    return out


def _benchmarks(raw_points: list[_RawPoint], *, fee_bps: float, slippage_bps: float) -> dict[str, BenchmarkMetric]:
    if not raw_points:
        empty = BenchmarkMetric(cumulative_return=0.0, max_drawdown=0.0)
        return {"buy_and_hold": empty, "ma_strategy": empty}
    # buy and hold: always +1
    bh_daily = [p.ret_1d for p in raw_points]
    buy_hold = BenchmarkMetric(
        cumulative_return=_cumulative_return(bh_daily),
        max_drawdown=_max_drawdown(bh_daily),
        sharpe_annualized_252=_sharpe_annualized_252(bh_daily),
        gross_strategy_return=sum(bh_daily),
        net_strategy_return=sum(bh_daily),
    )
    # MA strategy: +1 if ma20>ma60 else -1
    cost_rate = (fee_bps + slippage_bps) * 0.0001
    prev_pos = 0.0
    ma_daily_net: list[float] = []
    ma_daily_gross: list[float] = []
    for p in raw_points:
        pos = 1.0 if p.factors.ma20 > p.factors.ma60 else -1.0
        turnover = abs(pos - prev_pos)
        gross = pos * p.ret_1d
        net = gross - turnover * cost_rate
        ma_daily_gross.append(gross)
        ma_daily_net.append(net)
        prev_pos = pos
    ma_metric = BenchmarkMetric(
        cumulative_return=_cumulative_return(ma_daily_net),
        max_drawdown=_max_drawdown(ma_daily_net),
        sharpe_annualized_252=_sharpe_annualized_252(ma_daily_net),
        gross_strategy_return=sum(ma_daily_gross),
        net_strategy_return=sum(ma_daily_net),
    )
    return {"buy_and_hold": buy_hold, "ma_strategy": ma_metric}


def _valuation_backtest(
    *,
    raw_points: list[_RawPoint],
    data: UnifiedGoldData,
    settings: AppConfig,
    fee_bps: float,
    slippage_bps: float,
    position_scheme: str,
    score_bins_count: int,
) -> dict[str, Any]:
    frequency, coefficients = _resolve_backtest_pricing_coefficients(settings)
    base_payload = {
        "available": False,
        "with_valuation": None,
        "frequency": frequency,
    }
    if not raw_points:
        return {
            "valuation_metrics": {"available": False, "reason": "no backtest points"},
            "valuation_score_bins": [],
            "model_comparison": base_payload,
        }
    if not coefficients:
        return {
            "valuation_metrics": {
                "available": False,
                "reason": "pricing model coefficients unavailable",
                "frequency": frequency,
            },
            "valuation_score_bins": [],
            "model_comparison": base_payload,
        }

    scored: list[tuple[_RawPoint, int]] = []
    for point in raw_points:
        dxy = _latest_bar_close_until(data.dxy_daily, point.date)
        cb = _latest_obs_value_until(data.central_bank_gold_purchase_history, point.date)
        debt = _latest_obs_value_until(data.us_debt_history, point.date)
        inflation_multiplier = _inflation_multiplier_until(
            data.inflation_index_history, point.date
        )
        if dxy is None or cb is None or debt is None:
            continue
        valuation = calculate_pricing_model(
            PricingModelInput(
                gold_price=point.close,
                dxy_price=dxy,
                central_bank_gold_purchase_tonnes=cb,
                us_total_public_debt=debt,
                inflation_multiplier=inflation_multiplier,
                frequency=frequency,
            ),
            coefficients,
        )
        if valuation is None:
            continue
        scored.append((point, valuation.valuation_score))

    if not scored:
        return {
            "valuation_metrics": {
                "available": False,
                "reason": "valuation inputs unavailable for historical points",
                "frequency": frequency,
            },
            "valuation_score_bins": [],
            "model_comparison": base_payload,
        }

    enhanced_points: list[_RawPoint] = []
    for point, valuation_score in scored:
        factors = point.factors.model_copy(update={"valuation_score": valuation_score})
        enhanced_points.append(
            _RawPoint(
                date=point.date,
                close=point.close,
                score=calculate_factor_score(factors),
                factors=factors,
                direction_1d=point.direction_1d,
                ret_1d=point.ret_1d,
                forward_returns=point.forward_returns,
            )
        )
    enhanced_strategy = _compute_strategy_returns(
        enhanced_points,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        position_scheme=position_scheme,
    )
    enhanced_summary = _summarize(_to_points(enhanced_points, enhanced_strategy))
    valuation_scores = [score for _, score in scored]
    return {
        "valuation_metrics": {
            "available": True,
            "frequency": frequency,
            "count": len(scored),
            "avg_valuation_score": sum(valuation_scores) / len(valuation_scores),
            "min_valuation_score": min(valuation_scores),
            "max_valuation_score": max(valuation_scores),
        },
        "valuation_score_bins": _valuation_score_bin_metrics(
            scored, bins_count=score_bins_count
        ),
        "model_comparison": {
            "available": True,
            "frequency": frequency,
            "with_valuation": enhanced_summary.model_dump(),
        },
    }


def _resolve_backtest_pricing_coefficients(
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
    cached: dict[str, dict[str, Any]], frequency: str
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


def _valuation_score_bin_metrics(
    scored: list[tuple[_RawPoint, int]], *, bins_count: int
) -> list[ScoreBinMetric]:
    if not scored:
        return []
    pseudo_points: list[_RawPoint] = []
    for point, valuation_score in scored:
        score = point.score.model_copy(update={"total_score": valuation_score})
        pseudo_points.append(
            _RawPoint(
                date=point.date,
                close=point.close,
                score=score,
                factors=point.factors,
                direction_1d=point.direction_1d,
                ret_1d=point.ret_1d,
                forward_returns=point.forward_returns,
            )
        )
    return _score_bin_metrics(pseudo_points, bins_count=bins_count)


def _latest_bar_close_until(
    bars: list[UnifiedDailyBar], as_of: str
) -> float | None:
    values = [bar.close for bar in bars if bar.date <= as_of]
    return values[-1] if values else None


def _latest_obs_value_until(
    observations: list[UnifiedMacroObservation], as_of: str
) -> float | None:
    values = [
        item.value
        for item in sorted(observations, key=lambda x: x.date)
        if item.date <= as_of and item.value is not None
    ]
    return float(values[-1]) if values else None


def _inflation_multiplier_until(
    observations: list[UnifiedMacroObservation], as_of: str
) -> float:
    values = [
        item.value
        for item in sorted(observations, key=lambda x: x.date)
        if item.date <= as_of and item.value is not None and item.value > 0
    ]
    if len(values) < 2:
        return 1.0
    return float(values[-1]) / float(values[0])


def _rolling_window_mean(series: list[float], window: int) -> float | None:
    if not series:
        return None
    if len(series) < window:
        return sum(series) / len(series)
    vals = []
    for i in range(window - 1, len(series)):
        vals.append(sum(series[i - window + 1 : i + 1]) / window)
    return sum(vals) / len(vals) if vals else None


def _cumulative_return(daily_returns: list[float]) -> float:
    eq = 1.0
    for r in daily_returns:
        eq *= (1.0 + r)
    return eq - 1.0


def _max_drawdown(daily_returns: list[float]) -> float:
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in daily_returns:
        eq *= (1.0 + r)
        peak = max(peak, eq)
        dd = (eq / peak) - 1.0
        max_dd = min(max_dd, dd)
    return abs(max_dd)


def _sharpe_annualized_252(daily_returns: list[float]) -> float | None:
    if len(daily_returns) < 2:
        return None
    mean = sum(daily_returns) / len(daily_returns)
    std = pstdev(daily_returns)
    if std == 0:
        return None
    return (mean / std) * math.sqrt(252)


def _trend_from_bars_until(bars: list[UnifiedDailyBar], as_of: str) -> Literal["up", "flat", "down"] | None:
    values = [b.close for b in bars if b.date <= as_of]
    return _trend_from_values(values)


def _trend_from_obs_until(obs: list[UnifiedMacroObservation], as_of: str) -> Literal["up", "flat", "down"] | None:
    values = [x.value for x in obs if x.date <= as_of and x.value is not None]
    return _trend_from_values([float(v) for v in values])


def _trend_from_values(values: list[float], tolerance: float = 0.0) -> Literal["up", "flat", "down"] | None:
    if len(values) < 2:
        return None
    diff = values[-1] - values[-2]
    if abs(diff) <= tolerance:
        return "flat"
    return "up" if diff > 0 else "down"


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
