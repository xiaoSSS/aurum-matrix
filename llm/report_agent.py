"""LLM report generator with deterministic fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import requests
from pydantic import BaseModel

from backend.config import AppConfig, get_settings

REQUIRED_DECISION_FIELDS = (
    "macro_score",
    "technical_score",
    "flow_score",
    "total_score",
    "decision",
    "reasons",
    "risks",
)


def generate_markdown_report(
    decision_json: Mapping[str, Any] | BaseModel,
    *,
    settings: AppConfig | None = None,
) -> str:
    """Generate report via SiliconFlow LLM, fallback to deterministic template."""

    payload = _normalize_payload(decision_json)
    active_settings = settings or get_settings()
    try:
        return _generate_via_siliconflow(payload, active_settings)
    except Exception:
        return _generate_fallback_markdown(payload)


def _generate_via_siliconflow(payload: Mapping[str, Any], settings: AppConfig) -> str:
    api_key = settings.siliconflow_api_key
    if not api_key:
        raise ValueError("Missing siliconflow_api_key")

    url = settings.siliconflow_base_url.rstrip("/") + "/chat/completions"
    system_prompt = (
        "你是黄金投研报告助手。你只能依据输入JSON写中文Markdown，"
        "不得编造新数值。输出必须包含：当前结论、宏观环境、技术趋势、资金流、操作建议、风险提示。"
    )
    user_prompt = (
        "请基于以下结构化JSON生成报告：\n"
        f"{payload}\n"
        "注意：若某字段缺失，明确写“数据不足”。"
    )
    body = {
        "model": settings.siliconflow_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not content:
        raise ValueError("Empty LLM response content")
    return content


def _generate_fallback_markdown(payload: Mapping[str, Any]) -> str:
    missing_fields = [
        field for field in REQUIRED_DECISION_FIELDS if _is_empty(payload.get(field))
    ]
    macro_reasons = _section_items(payload, explicit_key="macro_reasons")
    technical_reasons = _section_items(payload, explicit_key="technical_reasons")
    flow_reasons = _section_items(payload, explicit_key="flow_reasons")
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
        lines.extend(["", f"> 数据不足：缺少或为空的字段：{', '.join(missing_fields)}。"])
    return "\n".join(lines)


def _normalize_payload(decision_json: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(decision_json, BaseModel):
        return decision_json.model_dump()
    if isinstance(decision_json, Mapping):
        return dict(decision_json)
    raise TypeError("decision_json must be mapping or Pydantic model")


def _conclusion(payload: Mapping[str, Any], missing_fields: Sequence[str]) -> str:
    if missing_fields:
        return "数据不足"
    return (
        f"当前结论：{payload['decision']}；总分：{payload['total_score']}；"
        f"宏观分：{payload['macro_score']}；技术分：{payload['technical_score']}；"
        f"资金流分：{payload['flow_score']}。"
    )


def _operation_advice(payload: Mapping[str, Any]) -> str:
    decision = payload.get("decision")
    if _is_empty(decision):
        return "数据不足"
    return f"操作建议：{decision}。本报告不提供收益承诺或确定性价格预测。"


def _score_line(label: str, value: Any) -> str:
    if _is_empty(value):
        return f"{label}：数据不足"
    return f"{label}：{value}"


def _section_items(payload: Mapping[str, Any], *, explicit_key: str) -> list[str]:
    explicit = _list_field(payload.get(explicit_key))
    if explicit:
        return explicit
    return _list_field(payload.get("reasons"))


def _bullet_block(items: Sequence[str]) -> str:
    if not items:
        return "- 数据不足"
    return "\n".join(f"- {item}" for item in items)


def _list_field(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return [str(item) for item in value if not _is_empty(item)]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return len(value) == 0
    return False
