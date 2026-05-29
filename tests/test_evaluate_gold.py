from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def test_evaluate_gold_endpoint_returns_structured_result() -> None:
    client = TestClient(app)

    response = client.get("/api/evaluate/gold")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XAUUSD"
    assert payload["price"] == 2368.4
    assert payload["currency"] == "USD/oz"
    assert payload["data_status"] == "ok"
    assert payload["decision"] == "观望"
    assert payload["score"]["total_score"] == 8
    assert payload["score"]["signal"] == "neutral"
    assert payload["technical_indicators"]["atr"] > 0
    assert "风险提示" in payload["risk_disclaimer"]
    assert "收益承诺" in payload["risk_disclaimer"]
