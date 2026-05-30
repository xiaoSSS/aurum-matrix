"""Optional LLM reporting package."""

from llm.report import generate_chinese_report
from llm.report_agent import generate_markdown_report

__all__ = ["generate_chinese_report", "generate_markdown_report"]
