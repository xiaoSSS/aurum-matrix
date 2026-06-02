from __future__ import annotations

from backend.config import AppConfig
from llm.report_agent import generate_markdown_report
from scoring.factor_score import FactorScoreInput, calculate_factor_score


def _decision_json() -> dict[str, object]:
    return calculate_factor_score(
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
    ).model_dump()


def test_generate_markdown_report_uses_llm_when_available(requests_mock) -> None:  # type: ignore[no-untyped-def]
    requests_mock.post(
        "https://api.siliconflow.cn/v1/chat/completions",
        json={"choices": [{"message": {"content": "# 测试报告\n\nLLM 输出"}}]},
    )
    report = generate_markdown_report(
        _decision_json(),
        settings=AppConfig(siliconflow_api_key="test-key"),
    )
    assert report.startswith("# 测试报告")


def test_generate_markdown_report_falls_back_when_llm_unavailable() -> None:
    report = generate_markdown_report(
        _decision_json(),
        settings=AppConfig(siliconflow_api_key=None),
    )
    assert "# 黄金当前价位评估报告" in report
    assert "## 当前结论" in report
    assert "## 风险提示" in report
