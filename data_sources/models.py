"""Typed data-source models for gold evaluation inputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class GoldMarketSnapshot(BaseModel):
    """Point-in-time market and macro data used by the evaluator."""

    symbol: Literal["XAUUSD"] = "XAUUSD"
    price: float = Field(gt=0, description="Gold spot price from the data source.")
    currency: Literal["USD/oz"] = "USD/oz"
    dxy: float = Field(gt=0, description="US Dollar Index value from the data source.")
    real_yield_10y: float = Field(description="10Y real yield from macro data source.")
    fed_funds_rate: float = Field(
        ge=0, description="Fed funds rate from macro data source."
    )
    etf_flow_tonnes_5d: float = Field(description="Five-day gold ETF flow in tonnes.")
    geopolitical_risk_index: float = Field(
        ge=0,
        le=100,
        description="Mock risk index where higher means stronger safe-haven demand.",
    )
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "mock"
