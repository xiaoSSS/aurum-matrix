"""FastAPI entry point for the gold price evaluation tool."""

from __future__ import annotations

from fastapi import FastAPI, Query

from backend.config import DataMode
from backend.services.gold_evaluator import EvaluateGoldResponse, GoldEvaluatorService

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
