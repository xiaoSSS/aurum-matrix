"""Deterministic report generation placeholder for future OpenAI integration."""

from __future__ import annotations

from scoring.models import GoldEvaluation


def generate_chinese_report(evaluation: GoldEvaluation) -> str:
    """Generate a Chinese report only from structured evaluation fields.

    This placeholder intentionally does not call an LLM. A future OpenAI-backed
    implementation must use these structured fields as the sole numeric source
    and must not invent prices, indicators, returns or positions.
    """

    if evaluation.data_status == "insufficient_data" or evaluation.score is None:
        return "数据不足，暂不生成配置或交易结论。请补充完整行情、宏观和技术指标数据。"
    return (
        f"黄金现价 {evaluation.price} {evaluation.currency}，"
        f"综合评分 {evaluation.score.total_score}，决策为{evaluation.decision}。"
        f"本结果用于当前价位评估，不构成收益承诺。"
    )
