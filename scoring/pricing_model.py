"""2025 CICC-style real gold fair-value pricing model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PricingFrequency = Literal["monthly", "quarterly", "annual"]


class PricingModelInput(BaseModel):
    """Inputs for the upgraded real-gold pricing model.

    The model predicts real gold price first, then converts it back to nominal
    price with ``inflation_multiplier``.
    """

    gold_price: float = Field(gt=0)
    dxy_price: float = Field(gt=0)
    central_bank_gold_purchase_tonnes: float
    us_total_public_debt: float = Field(gt=0)
    inflation_index: float | None = Field(default=None, gt=0)
    inflation_multiplier: float = Field(default=1.0, gt=0)
    frequency: PricingFrequency = "quarterly"


class PricingModelResult(BaseModel):
    frequency: PricingFrequency
    real_fair_value: float
    fair_value: float
    valuation_gap: float
    valuation_gap_pct: float
    valuation_state: str
    valuation_score: int
    factor_contributions: dict[str, float]


REQUIRED_COEFFICIENTS = {
    "intercept",
    "dxy_price",
    "central_bank_gold_purchase_tonnes",
    "us_total_public_debt",
}


def calculate_pricing_model(
    model_input: PricingModelInput, coefficients: dict[str, float]
) -> PricingModelResult | None:
    """Calculate nominal fair value and valuation score from fitted coefficients."""

    if not REQUIRED_COEFFICIENTS.issubset(coefficients):
        return None
    contributions = {
        "intercept": coefficients["intercept"],
        "central_bank_gold_purchase_tonnes": coefficients[
            "central_bank_gold_purchase_tonnes"
        ]
        * model_input.central_bank_gold_purchase_tonnes,
        "us_total_public_debt": coefficients["us_total_public_debt"]
        * model_input.us_total_public_debt,
        "dxy_price": coefficients["dxy_price"] * model_input.dxy_price,
    }
    real_fair_value = sum(contributions.values())
    fair_value = real_fair_value * model_input.inflation_multiplier
    if fair_value <= 0:
        return None

    gap = model_input.gold_price - fair_value
    gap_pct = gap / fair_value
    if gap_pct <= -0.05:
        state = "undervalued"
    elif gap_pct >= 0.05:
        state = "overvalued"
    else:
        state = "fair"
    score = _valuation_score(gap_pct)
    return PricingModelResult(
        frequency=model_input.frequency,
        real_fair_value=real_fair_value,
        fair_value=fair_value,
        valuation_gap=gap,
        valuation_gap_pct=gap_pct,
        valuation_state=state,
        valuation_score=score,
        factor_contributions=contributions,
    )


def select_pricing_coefficients(
    *,
    frequency: PricingFrequency,
    monthly: dict[str, float],
    quarterly: dict[str, float],
    annual: dict[str, float],
    legacy: dict[str, float] | None = None,
) -> tuple[PricingFrequency, dict[str, float]]:
    """Pick configured coefficients with quarterly-oriented fallback."""

    by_frequency: dict[PricingFrequency, dict[str, float]] = {
        "monthly": monthly,
        "quarterly": quarterly,
        "annual": annual,
    }
    fallback_order: dict[PricingFrequency, list[PricingFrequency]] = {
        "monthly": ["monthly", "quarterly", "annual"],
        "quarterly": ["quarterly", "monthly", "annual"],
        "annual": ["annual", "quarterly", "monthly"],
    }
    for candidate in fallback_order[frequency]:
        coefficients = by_frequency[candidate]
        if REQUIRED_COEFFICIENTS.issubset(coefficients):
            return candidate, coefficients
    if legacy and REQUIRED_COEFFICIENTS.issubset(legacy):
        return frequency, legacy
    return frequency, {}


def _valuation_score(gap_pct: float) -> int:
    """Map undervaluation to positive score and overvaluation to negative."""

    raw = -gap_pct * 500
    return int(max(-100, min(100, round(raw))))
