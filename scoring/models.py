"""Scoring models for gold evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from data_sources.models import GoldMarketSnapshot
from indicators.technical import TechnicalIndicators

ScoreLabel = Literal["bullish", "neutral", "bearish", "insufficient_data"]


class ScoreBreakdown(BaseModel):
    """Rule-based score details."""

    macro_score: int = Field(ge=-100, le=100)
    technical_score: int = Field(ge=-100, le=100)
    flow_score: int = Field(ge=-100, le=100)
    total_score: int = Field(ge=-100, le=100)
    signal: ScoreLabel
    factors: list[str]


class GoldEvaluation(BaseModel):
    """Structured response returned by the API and optional LLM module."""

    symbol: str
    price: float | None
    currency: str
    as_of: str
    data_status: Literal["ok", "insufficient_data"]
    decision: str
    score: ScoreBreakdown | None
    market_data: GoldMarketSnapshot | None
    technical_indicators: TechnicalIndicators | None
    summary: str
    risk_disclaimer: str
