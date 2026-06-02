from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.config import AppConfig, get_settings
from data_sources.fred_client import FredClient
from data_sources.market_client import AlphaVantageMarketClient


def test_app_config_reads_supported_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DATA_MODE", "mock")
    monkeypatch.setenv("MARKET_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("MACRO_PROVIDER", "fred")
    monkeypatch.setenv("FRED_API_KEY", "fred-key")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "av-key")
    monkeypatch.setenv("CACHE_DB_PATH", "cache.db")
    monkeypatch.setenv("DEFAULT_DXY_SYMBOL", "DXY_CUSTOM")

    settings = get_settings()

    assert settings.data_mode == "mock"
    assert settings.market_provider == "alpha_vantage"
    assert settings.macro_provider == "fred"
    assert settings.fred_api_key == "fred-key"
    assert settings.alpha_vantage_api_key == "av-key"
    assert settings.cache_db_path == Path("cache.db")
    assert settings.default_dxy_symbol == "DXY_CUSTOM"


def test_app_config_validates_supported_modes() -> None:
    with pytest.raises(ValidationError):
        AppConfig(data_mode="paper")  # type: ignore[arg-type]


def test_clients_accept_centralized_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = AppConfig(
        fred_api_key="fred-key",
        alpha_vantage_api_key="av-key",
        cache_db_path=tmp_path / "cache.db",
        default_dxy_symbol="DXY_CUSTOM",
    )

    fred_client = FredClient(settings=settings)
    market_client = AlphaVantageMarketClient(settings=settings)

    assert fred_client.api_key == "fred-key"
    assert fred_client.cache_path == tmp_path / "cache.db"
    assert market_client.api_key == "av-key"
    assert market_client.cache_path == tmp_path / "cache.db"
    assert market_client.dxy_symbol == "DXY_CUSTOM"


def test_app_config_reads_file() -> None:
    settings = AppConfig.from_file(
        Path(__file__).resolve().parents[1] / "aurum.config.example.json"
    )
    assert settings.market_provider == "alpha_vantage"
    assert settings.macro_provider == "fred"
    assert settings.default_dxy_symbol == "USDX"
