"""Training utilities for the upgraded CICC-style gold pricing model."""

from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from scoring.pricing_model import PricingFrequency


class PricingModelObservation(BaseModel):
    date: str
    gold_price: float = Field(gt=0)
    inflation_index: float = Field(gt=0)
    dxy_price: float = Field(gt=0)
    central_bank_gold_purchase_tonnes: float
    us_total_public_debt: float = Field(gt=0)


class PricingModelFitMetrics(BaseModel):
    r2: float | None = None
    mae: float | None = None
    rmse: float | None = None
    max_error: float | None = None
    sample_count: int = 0


class PricingModelFrequencyResult(BaseModel):
    frequency: PricingFrequency
    coefficients: dict[str, float] = Field(default_factory=dict)
    r2: float | None = None
    mae: float | None = None
    rmse: float | None = None
    max_error: float | None = None
    sample_count: int = 0
    train_start: str
    train_end: str
    validation_mae: float | None = None
    validation_rmse: float | None = None
    validation_max_error: float | None = None
    validation_sample_count: int = 0
    errors: list[str] = Field(default_factory=list)


class PricingModelTrainingBundle(BaseModel):
    results: dict[PricingFrequency, PricingModelFrequencyResult]
    errors: list[str] = Field(default_factory=list)


FEATURE_COLUMNS = [
    "central_bank_gold_purchase_tonnes",
    "us_total_public_debt",
    "dxy_price",
]
COEFFICIENT_NAMES = [
    "intercept",
    "central_bank_gold_purchase_tonnes",
    "us_total_public_debt",
    "dxy_price",
]


def train_pricing_models(
    observations: Iterable[PricingModelObservation],
    *,
    train_start: str = "2000-01-01",
    train_end: str = "2020-12-31",
    validation_start: str = "2021-01-01",
    validation_end: str = "2025-12-31",
) -> PricingModelTrainingBundle:
    """Fit monthly, quarterly and annual real-gold OLS models."""

    base_frame = _observations_to_frame(observations)
    results: dict[PricingFrequency, PricingModelFrequencyResult] = {}
    errors: list[str] = []
    for frequency in ("monthly", "quarterly", "annual"):
        result = fit_pricing_model_frequency(
            base_frame,
            frequency=frequency,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
        )
        results[frequency] = result
        errors.extend(result.errors)
    return PricingModelTrainingBundle(
        results=results,
        errors=list(dict.fromkeys(errors)),
    )


def fit_pricing_model_frequency(
    frame: pd.DataFrame,
    *,
    frequency: PricingFrequency,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
) -> PricingModelFrequencyResult:
    """Fit one frequency model from an already aligned daily/monthly frame."""

    result = PricingModelFrequencyResult(
        frequency=frequency,
        train_start=train_start,
        train_end=train_end,
    )
    if frame.empty:
        result.errors.append(f"{frequency}: no aligned observations")
        return result

    sampled = resample_pricing_frame(frame, frequency=frequency)
    if sampled.empty:
        result.errors.append(f"{frequency}: no observations after resample")
        return result

    train = sampled.loc[train_start:train_end].dropna()
    validation = sampled.loc[validation_start:validation_end].dropna()
    if len(train) < len(COEFFICIENT_NAMES):
        result.errors.append(
            f"{frequency}: insufficient training samples ({len(train)})"
        )
        result.sample_count = len(train)
        result.validation_sample_count = len(validation)
        return result

    coefficients = _ols_coefficients(train)
    train_pred = _predict(train, coefficients)
    train_metrics = _metrics(train["real_gold_price"].to_numpy(), train_pred)

    result.coefficients = {
        name: float(value) for name, value in zip(COEFFICIENT_NAMES, coefficients)
    }
    result.r2 = train_metrics.r2
    result.mae = train_metrics.mae
    result.rmse = train_metrics.rmse
    result.max_error = train_metrics.max_error
    result.sample_count = train_metrics.sample_count

    if not validation.empty:
        validation_pred = _predict(validation, coefficients)
        validation_metrics = _metrics(
            validation["real_gold_price"].to_numpy(), validation_pred
        )
        result.validation_mae = validation_metrics.mae
        result.validation_rmse = validation_metrics.rmse
        result.validation_max_error = validation_metrics.max_error
        result.validation_sample_count = validation_metrics.sample_count
    return result


def resample_pricing_frame(
    frame: pd.DataFrame, *, frequency: PricingFrequency
) -> pd.DataFrame:
    """Resample aligned observations to month, quarter or year end."""

    if frame.empty:
        return frame.copy()
    rule = {"monthly": "ME", "quarterly": "QE", "annual": "YE"}[frequency]
    sampled = frame.sort_index().resample(rule).last()
    return sampled.dropna(
        subset=[
            "gold_price",
            "inflation_index",
            "dxy_price",
            "central_bank_gold_purchase_tonnes",
            "us_total_public_debt",
        ]
    ).assign(
        inflation_multiplier=lambda x: x["inflation_index"] / x["inflation_index"].iloc[0],
        real_gold_price=lambda x: x["gold_price"] / x["inflation_multiplier"],
    )


def _observations_to_frame(
    observations: Iterable[PricingModelObservation],
) -> pd.DataFrame:
    rows = [item.model_dump() for item in observations]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    numeric_cols = [
        "gold_price",
        "inflation_index",
        "dxy_price",
        "central_bank_gold_purchase_tonnes",
        "us_total_public_debt",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _ols_coefficients(frame: pd.DataFrame) -> np.ndarray:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    ones = np.ones((len(x), 1), dtype=float)
    design = np.hstack([ones, x])
    y = frame["real_gold_price"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def _predict(frame: pd.DataFrame, coefficients: np.ndarray) -> np.ndarray:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    design = np.hstack([np.ones((len(x), 1), dtype=float), x])
    return design @ coefficients


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> PricingModelFitMetrics:
    if len(actual) == 0:
        return PricingModelFitMetrics()
    residuals = actual - predicted
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(sqrt(float(np.mean(residuals**2))))
    max_error = float(np.max(np.abs(residuals)))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    residual_sum = float(np.sum(residuals**2))
    r2 = None if total == 0 else 1.0 - residual_sum / total
    return PricingModelFitMetrics(
        r2=float(r2) if r2 is not None else None,
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        sample_count=int(len(actual)),
    )
