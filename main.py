"""FastAPI entry point for the gold price evaluation tool."""

from __future__ import annotations

import sqlite3

from fastapi import FastAPI, Query

from backend.config import DataMode, get_settings
from backend.services.backtest import BacktestResult, GoldBacktestService
from backend.services.gold_evaluator import EvaluateGoldResponse, GoldEvaluatorService
from backend.services.pricing_model_service import (
    PricingModelService,
    PricingModelStatusResponse,
    PricingModelTrainResponse,
)
from data_sources.sqlite_cache import initialize_time_series_cache

app = FastAPI(
    title="Aurum Matrix",
    description="黄金当前价位评估工具（真实数据源接入版本）",
    version="0.1.0",
)


@app.get("/api/evaluate/gold", response_model=EvaluateGoldResponse)
def evaluate_gold(
    mode: DataMode | None = Query(
        default=None, description="Override DATA_MODE for this request: mock or real."
    ),
) -> EvaluateGoldResponse:
    """Evaluate gold using the service layer only."""

    return GoldEvaluatorService().evaluate(mode=mode)


@app.get("/api/backtest/gold", response_model=BacktestResult)
def backtest_gold(
    mode: DataMode | None = Query(
        default=None, description="Override DATA_MODE for this request: mock or real."
    ),
    max_points: int = Query(default=250, ge=1, le=2000),
    horizons: str = Query(default="1,3,5,10,20"),
    fee_bps: float = Query(default=0.0, ge=0.0, le=1000.0),
    slippage_bps: float = Query(default=0.0, ge=0.0, le=1000.0),
    position_scheme: str = Query(default="strong1_weak0.5"),
) -> BacktestResult:
    """Run historical backtest of the current factor scoring model."""

    horizons_list = [int(x.strip()) for x in horizons.split(",") if x.strip()]
    return GoldBacktestService().run(
        mode=mode,
        max_points=max_points,
        horizons=horizons_list,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        position_scheme=position_scheme,
    )


@app.get("/api/backtest/gold/experiment")
def backtest_gold_experiment(
    mode: DataMode | None = Query(
        default=None, description="Override DATA_MODE for this request: mock or real."
    ),
    horizons: str = Query(default="1,3,5,10,20"),
    fee_bps: float = Query(default=0.0, ge=0.0, le=1000.0),
    slippage_bps: float = Query(default=0.0, ge=0.0, le=1000.0),
    fee_grid_bps: str = Query(default=""),
    slippage_grid_bps: str = Query(default=""),
    train_size: int = Query(default=120, ge=30, le=2000),
    test_size: int = Query(default=20, ge=5, le=1000),
    step_size: int = Query(default=20, ge=1, le=1000),
    position_scheme: str = Query(default="strong1_weak0.5"),
):
    """Run walk-forward style backtest experiment."""

    horizons_list = [int(x.strip()) for x in horizons.split(",") if x.strip()]
    fee_grid = [float(x.strip()) for x in fee_grid_bps.split(",") if x.strip()]
    slippage_grid = [float(x.strip()) for x in slippage_grid_bps.split(",") if x.strip()]
    return GoldBacktestService().run_experiment(
        mode=mode,
        horizons=horizons_list,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        fee_grid_bps=fee_grid or None,
        slippage_grid_bps=slippage_grid or None,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        position_scheme=position_scheme,
    )


@app.get("/api/data/status")
def data_status() -> dict[str, object]:
    """Return cache status for external data sources."""

    settings = get_settings()
    if settings.cache_db_path is None:
        return {"cache_enabled": False, "series": []}
    with sqlite3.connect(settings.cache_db_path) as connection:
        initialize_time_series_cache(connection)
        rows = connection.execute(
            """
            SELECT source, series_id, COUNT(*) AS rows_count, MAX(fetched_at)
            FROM time_series_cache
            GROUP BY source, series_id
            ORDER BY source, series_id
            """
        ).fetchall()
    return {
        "cache_enabled": True,
        "cache_db_path": str(settings.cache_db_path),
        "series": [
            {
                "source": source,
                "series_id": series_id,
                "rows": rows_count,
                "last_fetched_at": last_fetched_at,
                "available": rows_count > 0,
            }
            for source, series_id, rows_count, last_fetched_at in rows
        ],
    }


@app.post("/api/pricing-model/train", response_model=PricingModelTrainResponse)
def train_pricing_model() -> PricingModelTrainResponse:
    """Train monthly, quarterly and annual upgraded pricing models."""

    return PricingModelService().train()


@app.get("/api/pricing-model/status", response_model=PricingModelStatusResponse)
def pricing_model_status() -> PricingModelStatusResponse:
    """Return configured and cached pricing-model coefficient status."""

    return PricingModelService().status()


@app.post("/api/cache/refresh")
def refresh_cache(mode: DataMode | None = None) -> dict[str, object]:
    """Refresh cache by running the gold data provider once."""

    result = GoldEvaluatorService().evaluate(mode=mode)
    return {
        "status": "ok",
        "data_status": result.data_status,
        "enhanced_data_status": result.enhanced_data_status,
        "warnings": result.warnings,
        "errors": result.errors,
    }
