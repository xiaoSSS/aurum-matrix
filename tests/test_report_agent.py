from __future__ import annotations

from llm.report_agent import generate_markdown_report
from scoring.factor_score import FactorScoreInput, calculate_factor_score


def _decision_json() -> dict[str, object]:
    result = calculate_factor_score(
        FactorScoreInput(
            gold_price=2400.0,
            ma20=2380.0,
            ma60=2350.0,
            ma120=2300.0,
            macd_value=12.0,
            macd_signal=8.0,
            rsi=55.0,
            real_yield_trend="down",
            dxy_trend="down",
            etf_flow_trend="up",
            cftc_position_state="net_long_increasing",
        )
    )
    return result.model_dump()


def test_generate_markdown_report_contains_required_sections() -> None:
    report = generate_markdown_report(_decision_json())

    assert "# 黄金当前价位评估报告" in report
    assert "## 当前结论" in report
    assert "## 宏观环境" in report
    assert "## 技术趋势" in report
    assert "## 资金流" in report
    assert "## 操作建议" in report
    assert "## 风险提示" in report
    assert "当前结论：强多" in report
    assert "总分：64" in report
    assert "宏观分：65" in report
    assert "技术分：65" in report
    assert "资金流分：60" in report
    assert "实际利率趋势下行" in report
    assert "MACD 高于信号线" in report
    assert "黄金 ETF 资金流趋势改善" in report
    assert "不提供收益承诺" in report


def test_generate_markdown_report_uses_explicit_section_reasons() -> None:
    payload = {
        "macro_score": 10,
        "technical_score": 20,
        "flow_score": 30,
        "total_score": 20,
        "decision": "偏多",
        "reasons": ["通用原因"],
        "risks": ["已有风险提示"],
        "macro_reasons": ["宏观字段来自结构化 JSON"],
        "technical_reasons": ["技术字段来自结构化 JSON"],
        "flow_reasons": ["资金流字段来自结构化 JSON"],
    }

    report = generate_markdown_report(payload)

    assert "宏观字段来自结构化 JSON" in report
    assert "技术字段来自结构化 JSON" in report
    assert "资金流字段来自结构化 JSON" in report
    assert "已有风险提示" in report


def test_generate_markdown_report_marks_empty_fields_as_insufficient_data() -> None:
    payload = {
        "macro_score": None,
        "technical_score": 20,
        "flow_score": 0,
        "total_score": "",
        "decision": "",
        "reasons": [],
        "risks": [],
    }

    report = generate_markdown_report(payload)

    assert "当前结论\n数据不足" in report
    assert "宏观分：数据不足" in report
    assert "## 技术趋势\n技术分：20\n- 数据不足" in report
    assert "## 资金流\n资金流分：0\n- 数据不足" in report
    assert "## 操作建议\n数据不足" in report
    assert "## 风险提示\n- 数据不足" in report
    assert "缺少或为空的字段" in report


def test_generate_markdown_report_accepts_pydantic_result() -> None:
    result = calculate_factor_score(
        FactorScoreInput(
            gold_price=2200.0,
            ma20=2250.0,
            ma60=2300.0,
            ma120=2350.0,
            macd_value=-15.0,
            macd_signal=-8.0,
            rsi=82.0,
            real_yield_trend="up",
            dxy_trend="up",
            etf_flow_trend="down",
            cftc_position_state="net_short",
        )
    )

    report = generate_markdown_report(result)

    assert "当前结论：强空" in report
    assert "总分：-68" in report
    assert "实际利率趋势上行" in report
    assert "均线空头排列" in report
    assert "ETF 资金持续流出" in report
