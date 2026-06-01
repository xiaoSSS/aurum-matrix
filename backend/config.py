"""Centralized runtime configuration for Aurum Matrix.

Configuration is intentionally read in one place so data clients do not scatter
``os.environ`` access across the codebase. Values can still be overridden by
explicit constructor arguments in tests or custom integrations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DataMode = Literal["mock", "real"]
MarketProvider = Literal["alpha_vantage"]
MacroProvider = Literal["fred"]


class AppConfig(BaseModel):
    """Application settings loaded from environment variables."""

    data_mode: DataMode = "real"
    market_provider: MarketProvider = "alpha_vantage"
    macro_provider: MacroProvider = "fred"
    fred_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    cache_db_path: Path | None = None
    default_dxy_symbol: str = Field(default="DXY", min_length=1)

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
            default_dxy_symbol=os.environ.get("DEFAULT_DXY_SYMBOL", "DXY"),
        )


def get_settings() -> AppConfig:
    """Return current application settings from the process environment."""

    return AppConfig.from_env()
