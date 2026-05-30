"""Decision rules for converting scores into portfolio actions."""

from __future__ import annotations

from scoring.models import ScoreBreakdown


def decide_gold_action(score: ScoreBreakdown | None) -> str:
    """Return 增配/持有/观望/减配/回避, or 数据不足 when inputs are missing."""

    if score is None or score.signal == "insufficient_data":
        return "数据不足"
    if score.total_score >= 45:
        return "增配"
    if score.total_score >= 15:
        return "持有"
    if score.total_score > -15:
        return "观望"
    if score.total_score > -45:
        return "减配"
    return "回避"
