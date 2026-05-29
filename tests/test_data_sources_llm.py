from __future__ import annotations

import json

from data_sources.mock_gold import get_mock_gold_snapshot
from data_sources.sqlite_cache import open_sqlite_connection
from llm.report import generate_chinese_report
from scoring.models import GoldEvaluation


def test_mock_snapshot_contains_source_values() -> None:
    snapshot = get_mock_gold_snapshot()

    assert snapshot.symbol == "XAUUSD"
    assert snapshot.price == 2368.4
    assert snapshot.source == "mock"


def test_sqlite_cache_initializes_snapshot_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "aurum.db"

    connection = open_sqlite_connection(database_path)
    try:
        connection.execute(
            "INSERT INTO market_snapshots (symbol, source, payload_json) VALUES (?, ?, ?)",
            ("XAUUSD", "mock", json.dumps({"price": 2368.4})),
        )
        count = connection.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[
            0
        ]
    finally:
        connection.close()

    assert count == 1


def test_report_returns_insufficient_data_without_inventing_numbers() -> None:
    evaluation = GoldEvaluation(
        symbol="XAUUSD",
        price=None,
        currency="USD/oz",
        as_of="2026-05-29T00:00:00+00:00",
        data_status="insufficient_data",
        decision="数据不足",
        score=None,
        market_data=None,
        technical_indicators=None,
        summary="数据不足，无法完成黄金当前价位评估。",
        risk_disclaimer="风险提示：不提供收益承诺。",
    )

    report = generate_chinese_report(evaluation)

    assert "数据不足" in report
    assert "2368" not in report
