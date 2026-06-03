from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def test_evaluate_gold_endpoint_returns_insufficient_without_live_credentials(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("DATA_MODE", "real")
    client = TestClient(app)

    response = client.get("/api/evaluate/gold")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XAUUSD"
    assert payload["price"] is None
    assert payload["currency"] == "USD/oz"
    assert payload["data_status"] == "insufficient_data"
    assert payload["market_state"] == "数据不足"
    assert payload["action"] == "数据不足"
    assert payload["score"] is None
    assert payload["factor_input"] is None
    assert payload["indicators"] is None
    assert "数据不足" in payload["report_markdown"]
    assert "风险提示" in payload["risk_disclaimer"]
    assert "收益承诺" in payload["risk_disclaimer"]


def test_evaluate_gold_endpoint_supports_mode_query_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("DATA_MODE", "real")
    client = TestClient(app)

    response = client.get("/api/evaluate/gold?mode=mock")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_mode"] == "mock"
    assert payload["provider"] == "mock"
    assert payload["price"] == 2368.4


def test_pricing_model_status_endpoint_returns_shape() -> None:
    client = TestClient(app)

    response = client.get("/api/pricing-model/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured_frequency"] == "quarterly"
    assert "configured_coefficients" in payload
    assert "cached_coefficients" in payload


def test_pricing_model_train_endpoint_degrades_without_live_data(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    client = TestClient(app)

    response = client.post("/api/pricing-model/train")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "insufficient_data"}
    assert "results" in payload
    assert "errors" in payload
