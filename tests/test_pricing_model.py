from __future__ import annotations

from datetime import date, timedelta

from backend.config import AppConfig
from backend.services.data_provider import UnifiedDailyBar, UnifiedMacroObservation
from backend.services.pricing_model_service import (
    PricingModelService,
    align_pricing_observations,
)
from data_sources.sqlite_cache import (
    read_pricing_model_coefficients,
    write_pricing_model_coefficients,
)
from scoring.pricing_model import PricingModelInput, calculate_pricing_model
from scoring.pricing_model_training import (
    PricingModelObservation,
    train_pricing_models,
)


def test_pricing_model_calculates_real_and_nominal_fair_value() -> None:
    result = calculate_pricing_model(
        PricingModelInput(
            gold_price=1900,
            dxy_price=100,
            central_bank_gold_purchase_tonnes=50,
            us_total_public_debt=34000,
            inflation_index=200,
            inflation_multiplier=2.0,
            frequency="quarterly",
        ),
        {
            "intercept": 1000,
            "dxy_price": -4,
            "central_bank_gold_purchase_tonnes": 3,
            "us_total_public_debt": 0.01,
        },
    )
    assert result is not None
    assert result.frequency == "quarterly"
    assert result.real_fair_value == 1090
    assert result.fair_value == 2180
    assert result.valuation_gap == -280
    assert result.valuation_state == "undervalued"
    assert result.valuation_score > 0


def test_pricing_model_returns_none_when_coefficients_missing() -> None:
    result = calculate_pricing_model(
        PricingModelInput(
            gold_price=1900,
            dxy_price=100,
            central_bank_gold_purchase_tonnes=50,
            us_total_public_debt=34000,
        ),
        {"intercept": 2000},
    )
    assert result is None


def test_pricing_training_recovers_ols_coefficients() -> None:
    observations: list[PricingModelObservation] = []
    first = date(2000, 1, 1)
    for index in range(96):
        current = first + timedelta(days=31 * index)
        cb = 20 + index * 0.5
        debt = 5000 + index * 20
        dxy = 95 + (index % 8)
        inflation = 100 + index * 0.2
        real_gold = 700 + 2 * cb + 0.03 * debt - 1.5 * dxy
        gold = real_gold * (inflation / 100)
        observations.append(
            PricingModelObservation(
                date=current.isoformat(),
                gold_price=gold,
                inflation_index=inflation,
                dxy_price=dxy,
                central_bank_gold_purchase_tonnes=cb,
                us_total_public_debt=debt,
            )
        )

    bundle = train_pricing_models(
        observations,
        train_start="2000-01-01",
        train_end="2006-12-31",
        validation_start="2007-01-01",
        validation_end="2008-12-31",
    )

    monthly = bundle.results["monthly"]
    assert monthly.sample_count > 0
    assert monthly.validation_sample_count > 0
    assert monthly.coefficients["intercept"] == monthly.coefficients["intercept"]
    assert monthly.mae is not None
    assert monthly.rmse is not None
    assert "quarterly" in bundle.results
    assert "annual" in bundle.results


def test_align_pricing_observations_forward_fills_low_frequency_series() -> None:
    gold = [
        UnifiedDailyBar(date="2024-01-01", open=1, high=1, low=1, close=2000),
        UnifiedDailyBar(date="2024-01-02", open=1, high=1, low=1, close=2010),
    ]
    dxy = [
        UnifiedDailyBar(date="2024-01-01", open=1, high=1, low=1, close=100),
        UnifiedDailyBar(date="2024-01-02", open=1, high=1, low=1, close=101),
    ]
    observations = align_pricing_observations(
        gold=gold,
        dxy=dxy,
        inflation=[UnifiedMacroObservation(date="2024-01-01", value=300)],
        debt=[UnifiedMacroObservation(date="2024-01-01", value=34000)],
        central_bank_gold=[UnifiedMacroObservation(date="2024-01-01", value=40)],
        start="2024-01-01",
        end="2024-01-31",
    )
    assert len(observations) == 2
    assert observations[-1].dxy_price == 101
    assert observations[-1].us_total_public_debt == 34000


def test_pricing_model_status_reads_cached_coefficients(tmp_path) -> None:
    cache = tmp_path / "cache.db"
    write_pricing_model_coefficients(
        cache,
        frequency="quarterly",
        coefficients={
            "intercept": 1,
            "central_bank_gold_purchase_tonnes": 2,
            "us_total_public_debt": 3,
            "dxy_price": 4,
        },
        metrics={"sample_count": 10},
        train_start="2000-01-01",
        train_end="2020-12-31",
    )
    cached = read_pricing_model_coefficients(cache)
    assert "quarterly" in cached

    status = PricingModelService(
        settings=AppConfig(cache_db_path=cache)
    ).status()
    assert status.cache_enabled is True
    assert "quarterly" in status.cached_coefficients
