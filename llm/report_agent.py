"""Deterministic Markdown report agent for structured decision-engine output.

This module intentionally does not call an LLM. It only formats values already
present in the decision-engine JSON and uses ``数据不足`` whenever required
fields or section details are missing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

REQUIRED_DECISION_FIELDS = (
    "macro_score",
    "technical_score",
    "flow_score",
    "total_score",
    "decision",
    "reasons",
    "risks",
)

MACRO_KEYWORDS = ("实际利率", "美元", "DXY", "宏观", "通胀")
TECHNICAL_KEYWORDS = ("金价", "MA", "均线", "MACD", "RSI", "技术", "趋势")
FLOW_KEYWORDS = ("ETF", "CFTC", "资金", "持仓", "投机")


def generate_markdown_report(decision_json: Mapping[str, Any] | BaseModel) -> str:
    """Generate a Chinese Markdown report from structured decision JSON only.

    The function does not infer or create new numeric values. Every score and
    text fragment in the report is copied from the supplied structured input.
    Missing or empty fields are rendered as ``数据不足``.
    """

    payload = _normalize_payload(decision_json)
    missing_fields = [
        field for field in REQUIRED_DECISION_FIELDS if _is_empty(payload.get(field))
    ]

    macro_reasons = _section_items(
        payload, explicit_key="macro_reasons", keywords=MACRO_KEYWORDS
    )
    technical_reasons = _section_items(
        payload,
        explicit_key="technical_reasons",
        keywords=TECHNICAL_KEYWORDS,
    )
    flow_reasons = _section_items(
        payload, explicit_key="flow_reasons", keywords=FLOW_KEYWORDS
    )
    risks = _list_field(payload.get("risks"))

    lines = ["# 黄金当前价位评估报告", ""]
    lines.extend(
        [
            "## 当前结论",
            _conclusion(payload, missing_fields),
            "",
            "## 宏观环境",
            _score_line("宏观分", payload.get("macro_score")),
            _bullet_block(macro_reasons),
            "",
            "## 技术趋势",
            _score_line("技术分", payload.get("technical_score")),
            _bullet_block(technical_reasons),
            "",
            "## 资金流",
            _score_line("资金流分", payload.get("flow_score")),
            _bullet_block(flow_reasons),
            "",
            "## 操作建议",
            _operation_advice(payload),
            "",
            "## 风险提示",
            _bullet_block(risks),
        ]
    )
    if missing_fields:
        lines.extend(
            ["", f"> 数据不足：缺少或为空的字段：{', '.join(missing_fields)}。"]
        )
    return "\n".join(lines)


def _normalize_payload(decision_json: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    """Normalize a mapping or Pydantic model to a plain dict."""

    if isinstance(decision_json, BaseModel):
        return decision_json.model_dump()
    if isinstance(decision_json, Mapping):
        return dict(decision_json)
    raise TypeError("decision_json must be a structured JSON mapping or Pydantic model")


def _conclusion(payload: Mapping[str, Any], missing_fields: Sequence[str]) -> str:
    """Build the conclusion line using only provided values."""

    if missing_fields:
        return "数据不足"
    return (
        f"当前结论：{payload['decision']}；"
        f"总分：{payload['total_score']}；"
        f"宏观分：{payload['macro_score']}；"
        f"技术分：{payload['technical_score']}；"
        f"资金流分：{payload['flow_score']}。"
    )


def _operation_advice(payload: Mapping[str, Any]) -> str:
    """Build operation advice without adding numbers or unsupported projections."""

    decision = payload.get("decision")
    if _is_empty(decision):
        return "数据不足"
    return f"操作建议：{decision}。本报告不提供收益承诺或确定性价格预测。"


def _score_line(label: str, value: Any) -> str:
    """Render a score line or 数据不足 when the score is missing."""

    if _is_empty(value):
        return f"{label}：数据不足"
    return f"{label}：{value}"


def _section_items(
    payload: Mapping[str, Any], *, explicit_key: str, keywords: Sequence[str]
) -> list[str]:
    """Return explicit section items, or categorize general reasons by keywords."""

    explicit_items = _list_field(payload.get(explicit_key))
    if explicit_items:
        return explicit_items

    reasons = _list_field(payload.get("reasons"))
    return [
        reason for reason in reasons if any(keyword in reason for keyword in keywords)
    ]


def _bullet_block(items: Sequence[str]) -> str:
    """Render bullet list or 数据不足 for an empty section."""

    if not items:
        return "- 数据不足"
    return "\n".join(f"- {item}" for item in items)


def _list_field(value: Any) -> list[str]:
    """Return a clean list of non-empty strings from a JSON array-like value."""

    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return [str(item) for item in value if not _is_empty(item)]


def _is_empty(value: Any) -> bool:
    """Return True for missing, blank, or empty collection values."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return len(value) == 0
    return False
