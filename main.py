"""FastAPI entry point for the gold price evaluation tool."""

from __future__ import annotations

from fastapi import FastAPI

from data_sources.mock_gold import get_mock_gold_price_history, get_mock_gold_snapshot
from decision_engine import decide_gold_action
from indicators.technical import calculate_technical_indicators
from llm import generate_chinese_report
from scoring.gold import calculate_gold_score
from scoring.models import GoldEvaluation

RISK_DISCLAIMER = "风险提示：本工具仅用于黄金当前价位评估，不提供确定性价格预测或收益承诺，交易需自行承担风险。"

app = FastAPI(
    title="Aurum Matrix",
    description="黄金当前价位评估工具（mock 数据版本）",
    version="0.1.0",
)


@app.get("/api/evaluate/gold", response_model=GoldEvaluation)
def evaluate_gold() -> GoldEvaluation:
    """Evaluate the current gold price using deterministic mock data."""

    snapshot = get_mock_gold_snapshot()
    history = get_mock_gold_price_history()
    technical = calculate_technical_indicators(history)
    if technical is None:
        return GoldEvaluation(
            symbol="XAUUSD",
            price=None,
            currency="USD/oz",
            as_of=snapshot.as_of.isoformat(),
            data_status="insufficient_data",
            decision="数据不足",
            score=None,
            market_data=snapshot,
            technical_indicators=None,
            summary="数据不足，无法完成黄金当前价位评估。",
            risk_disclaimer=RISK_DISCLAIMER,
        )

    score = calculate_gold_score(snapshot, technical)
    decision = decide_gold_action(score)
    evaluation = GoldEvaluation(
        symbol=snapshot.symbol,
        price=snapshot.price,
        currency=snapshot.currency,
        as_of=snapshot.as_of.isoformat(),
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
