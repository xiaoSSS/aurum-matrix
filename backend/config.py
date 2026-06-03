"""Centralized runtime configuration for Aurum Matrix.

Configuration is intentionally read in one place so data clients do not scatter
settings-loading logic across the codebase.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

DataMode = Literal["mock", "real"]
MarketProvider = Literal["alpha_vantage"]
MacroProvider = Literal["fred"]
PricingModelFrequency = Literal["monthly", "quarterly", "annual"]


class AppConfig(BaseModel):
    """Application settings loaded from a config file and optional env overrides."""

    model_config = ConfigDict(extra="ignore")

    data_mode: DataMode = "real"
    market_provider: MarketProvider = "alpha_vantage"
    macro_provider: MacroProvider = "fred"
    fred_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    cache_db_path: Path | None = None
    default_dxy_symbol: str = Field(default="USDX", min_length=1)
    siliconflow_api_key: str | None = None
    siliconflow_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"

    cache_ttl_market_hours: int = 24
    cache_ttl_macro_hours: int = 24
    cache_ttl_cftc_hours: int = 24
    cache_ttl_etf_hours: int = 24
    cache_ttl_central_bank_gold_hours: int = 720

    pricing_model_coefficients: dict[str, float] = Field(default_factory=dict)
    pricing_model_coefficients_monthly: dict[str, float] = Field(default_factory=dict)
    pricing_model_coefficients_quarterly: dict[str, float] = Field(default_factory=dict)
    pricing_model_coefficients_annual: dict[str, float] = Field(default_factory=dict)
    pricing_model_frequency: PricingModelFrequency = "quarterly"
    pricing_model_train_start: str = "2000-01-01"
    pricing_model_train_end: str = "2025-12-31"
    pricing_model_validation_start: str = "2021-01-01"
    pricing_model_validation_end: str = "2025-12-31"
    pricing_model_inflation_series: str = "CPIAUCSL"
    etf_flows_url: str | None = None
    cftc_cot_url: str | None = None
    central_bank_gold_url: str | None = None
    imf_gold_reserves_url: str | None = None

    @classmethod
    def from_file(cls, path: Path | None = None) -> "AppConfig":
        """Load settings from a JSON config file.

        Missing file falls back to defaults.
        """

        env_config_path = os.environ.get("AURUM_CONFIG_FILE")
        config_path = (
            path
            or (Path(env_config_path) if env_config_path else None)
            or Path(__file__).resolve().parents[1] / "aurum.config.json"
        )
        if not config_path.exists():
            return cls()

        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must contain a JSON object: {config_path}")
        return cls.model_validate(raw)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load settings from environment variables."""

        cache_path = os.environ.get("CACHE_DB_PATH") or os.environ.get(
            "AURUM_SQLITE_CACHE_PATH"
        )
        return cls(
            data_mode=os.environ.get("DATA_MODE", "real"),
            market_provider=os.environ.get("MARKET_PROVIDER", "alpha_vantage"),
            macro_provider=os.environ.get("MACRO_PROVIDER", "fred"),
            fred_api_key=os.environ.get("FRED_API_KEY"),
            alpha_vantage_api_key=os.environ.get("ALPHA_VANTAGE_API_KEY"),
            cache_db_path=Path(cache_path) if cache_path else None,
            default_dxy_symbol=os.environ.get("DEFAULT_DXY_SYMBOL", "USDX"),
            siliconflow_api_key=os.environ.get("SILICONFLOW_API_KEY"),
            siliconflow_model=os.environ.get(
                "SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V4-Pro"
            ),
            siliconflow_base_url=os.environ.get(
                "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
            ),
        )


def get_settings() -> AppConfig:
    """Return current application settings from file plus env overrides."""

    file_settings = AppConfig.from_file()
    cache_path = os.environ.get("CACHE_DB_PATH") or os.environ.get(
        "AURUM_SQLITE_CACHE_PATH"
    )
    env_updates: dict[str, object | None] = {}
    mapping = {
        "DATA_MODE": "data_mode",
        "MARKET_PROVIDER": "market_provider",
        "MACRO_PROVIDER": "macro_provider",
        "FRED_API_KEY": "fred_api_key",
        "ALPHA_VANTAGE_API_KEY": "alpha_vantage_api_key",
        "DEFAULT_DXY_SYMBOL": "default_dxy_symbol",
        "SILICONFLOW_API_KEY": "siliconflow_api_key",
        "SILICONFLOW_MODEL": "siliconflow_model",
        "SILICONFLOW_BASE_URL": "siliconflow_base_url",
        "ETF_FLOWS_URL": "etf_flows_url",
        "CFTC_COT_URL": "cftc_cot_url",
        "CENTRAL_BANK_GOLD_URL": "central_bank_gold_url",
        "IMF_GOLD_RESERVES_URL": "imf_gold_reserves_url",
        "PRICING_MODEL_FREQUENCY": "pricing_model_frequency",
        "PRICING_MODEL_TRAIN_START": "pricing_model_train_start",
        "PRICING_MODEL_TRAIN_END": "pricing_model_train_end",
        "PRICING_MODEL_VALIDATION_START": "pricing_model_validation_start",
        "PRICING_MODEL_VALIDATION_END": "pricing_model_validation_end",
        "PRICING_MODEL_INFLATION_SERIES": "pricing_model_inflation_series",
    }
    for env_name, field_name in mapping.items():
        if env_name in os.environ:
            env_updates[field_name] = os.environ.get(env_name)
    if cache_path:
        env_updates["cache_db_path"] = Path(cache_path)
    if not env_updates:
        return file_settings
    return file_settings.model_copy(update=env_updates)
