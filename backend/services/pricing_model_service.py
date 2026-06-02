"""Service layer for fitting and reading upgraded gold pricing models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import BaseModel, Field

from backend.config import AppConfig, get_settings
from backend.services.data_provider import UnifiedDailyBar, UnifiedMacroObservation
from data_sources.enhanced_clients import EnhancedDataClient
from data_sources.fred_client import FredClient, FredClientError
from data_sources.market_client import AlphaVantageMarketClient, MarketClientError
from data_sources.sqlite_cache import (
    read_pricing_model_coefficients,
    write_pricing_model_coefficients,
)
from scoring.pricing_model import PricingFrequency
from scoring.pricing_model_training import (
    PricingModelObservation,
    PricingModelTrainingBundle,
    train_pricing_models,
)


class PricingModelTrainResponse(BaseModel):
    status: str
    results: dict[PricingFrequency, object] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class PricingModelStatusResponse(BaseModel):
    cache_enabled: bool
    configured_frequency: PricingFrequency
    configured_coefficients: dict[str, bool]
    cached_coefficients: dict[str, object] = Field(default_factory=dict)


@dataclass
class _SeriesSet:
    gold: list[UnifiedDailyBar]
    dxy: list[UnifiedDailyBar]
    inflation: list[UnifiedMacroObservation]
    debt: list[UnifiedMacroObservation]
    central_bank_gold: list[UnifiedMacroObservation]
    errors: list[str]


class PricingModelService:
    """Train, persist and report status for pricing-model coefficients."""

    def __init__(self, *, settings: AppConfig | None = None) -> None:
        self.settings = settings or get_settings()

    def train(self) -> PricingModelTrainResponse:
        series = self._load_series()
        observations = align_pricing_observations(
            gold=series.gold,
            dxy=series.dxy,
            inflation=series.inflation,
            debt=series.debt,
            central_bank_gold=series.central_bank_gold,
            start=self.settings.pricing_model_train_start,
            end=self.settings.pricing_model_train_end,
        )
        if not observations:
            return PricingModelTrainResponse(
                status="insufficient_data",
                errors=[
                    *series.errors,
                    "No aligned observations for pricing-model training.",
                ],
            )

        bundle = train_pricing_models(
            observations,
            train_start=self.settings.pricing_model_train_start,
            train_end="2020-12-31",
            validation_start=self.settings.pricing_model_validation_start,
            validation_end=self.settings.pricing_model_validation_end,
        )
        persisted_errors = self._persist_successful_results(bundle)
        errors = list(dict.fromkeys([*series.errors, *bundle.errors, *persisted_errors]))
        ok_count = sum(1 for item in bundle.results.values() if item.coefficients)
        return PricingModelTrainResponse(
            status="ok" if ok_count else "insufficient_data",
            results={freq: result.model_dump() for freq, result in bundle.results.items()},
            errors=errors,
        )

    def status(self) -> PricingModelStatusResponse:
        cached: dict[str, object] = {}
        if self.settings.cache_db_path is not None:
            cached = read_pricing_model_coefficients(self.settings.cache_db_path)
        return PricingModelStatusResponse(
            cache_enabled=self.settings.cache_db_path is not None,
            configured_frequency=self.settings.pricing_model_frequency,
            configured_coefficients={
                "monthly": bool(self.settings.pricing_model_coefficients_monthly),
                "quarterly": bool(self.settings.pricing_model_coefficients_quarterly),
                "annual": bool(self.settings.pricing_model_coefficients_annual),
                "legacy": bool(self.settings.pricing_model_coefficients),
            },
            cached_coefficients=cached,
        )

    def _load_series(self) -> _SeriesSet:
        errors: list[str] = []
        gold: list[UnifiedDailyBar] = []
        dxy: list[UnifiedDailyBar] = []
        inflation: list[UnifiedMacroObservation] = []
        debt: list[UnifiedMacroObservation] = []
        central_bank_gold: list[UnifiedMacroObservation] = []

        try:
            market = AlphaVantageMarketClient(settings=self.settings)
            gold = [
                UnifiedDailyBar(**bar.model_dump())
                for bar in market.get_xau_usd_daily(outputsize="full").bars
            ]
            dxy = [
                UnifiedDailyBar(**bar.model_dump())
                for bar in market.get_dxy_daily(outputsize="full").bars
            ]
        except MarketClientError as exc:
            errors.append(f"Market history unavailable: {exc}")

        try:
            macro = FredClient(settings=self.settings)
            inflation_data = macro.get_cpi_index(
                series_id=self.settings.pricing_model_inflation_series,
                observation_start=self.settings.pricing_model_train_start,
                observation_end=self.settings.pricing_model_train_end,
            )
            debt_data = macro.get_us_total_public_debt(
                observation_start=self.settings.pricing_model_train_start,
                observation_end=self.settings.pricing_model_train_end,
            )
            inflation = [
                UnifiedMacroObservation(date=o.date, value=o.value)
                for o in inflation_data.observations
            ]
            debt = [
                UnifiedMacroObservation(date=o.date, value=o.value)
                for o in debt_data.observations
            ]
        except FredClientError as exc:
            errors.append(f"FRED history unavailable: {exc}")

        cb_series = EnhancedDataClient(
            source="wgc",
            series_id="central_bank_gold",
            url=self.settings.central_bank_gold_url or self.settings.imf_gold_reserves_url,
            cache_path=self.settings.cache_db_path,
            cache_ttl_hours=self.settings.cache_ttl_central_bank_gold_hours,
        ).get_series()
        errors.extend(cb_series.warnings)
        central_bank_gold = [
            UnifiedMacroObservation(date=o.date, value=o.value)
            for o in cb_series.observations
        ]
        return _SeriesSet(
            gold=gold,
            dxy=dxy,
            inflation=inflation,
            debt=debt,
            central_bank_gold=central_bank_gold,
            errors=list(dict.fromkeys(errors)),
        )

    def _persist_successful_results(
        self, bundle: PricingModelTrainingBundle
    ) -> list[str]:
        if self.settings.cache_db_path is None:
            return ["Pricing model coefficients not persisted: cache_db_path is not configured."]
        errors: list[str] = []
        cache_path = Path(self.settings.cache_db_path)
        for frequency, result in bundle.results.items():
            if not result.coefficients:
                continue
            try:
                write_pricing_model_coefficients(
                    cache_path,
                    frequency=frequency,
                    coefficients=result.coefficients,
                    metrics=result.model_dump(),
                    train_start=result.train_start,
                    train_end=result.train_end,
                )
            except OSError as exc:
                errors.append(f"Persist {frequency} coefficients failed: {exc}")
        return errors


def align_pricing_observations(
    *,
    gold: Iterable[UnifiedDailyBar],
    dxy: Iterable[UnifiedDailyBar],
    inflation: Iterable[UnifiedMacroObservation],
    debt: Iterable[UnifiedMacroObservation],
    central_bank_gold: Iterable[UnifiedMacroObservation],
    start: str,
    end: str,
) -> list[PricingModelObservation]:
    """Align mixed-frequency pricing inputs by forward-filling lower-frequency data."""

    gold_frame = _bars_to_frame(gold, "gold_price")
    dxy_frame = _bars_to_frame(dxy, "dxy_price")
    inflation_frame = _obs_to_frame(inflation, "inflation_index")
    debt_frame = _obs_to_frame(debt, "us_total_public_debt")
    cb_frame = _obs_to_frame(central_bank_gold, "central_bank_gold_purchase_tonnes")
    if gold_frame.empty:
        return []
    frame = gold_frame.join(dxy_frame, how="left")
    for extra in (inflation_frame, debt_frame, cb_frame):
        frame = frame.join(extra, how="left")
    frame = frame.sort_index().loc[start:end].ffill().dropna()
    return [
        PricingModelObservation(
            date=index.date().isoformat(),
            gold_price=float(row["gold_price"]),
            inflation_index=float(row["inflation_index"]),
            dxy_price=float(row["dxy_price"]),
            central_bank_gold_purchase_tonnes=float(
                row["central_bank_gold_purchase_tonnes"]
            ),
            us_total_public_debt=float(row["us_total_public_debt"]),
        )
        for index, row in frame.iterrows()
    ]


def _bars_to_frame(bars: Iterable[UnifiedDailyBar], column: str) -> pd.DataFrame:
    rows = [{"date": item.date, column: item.close} for item in bars]
    return _rows_to_frame(rows, column)


def _obs_to_frame(
    observations: Iterable[UnifiedMacroObservation], column: str
) -> pd.DataFrame:
    rows = [
        {"date": item.date, column: item.value}
        for item in observations
        if item.value is not None
    ]
    return _rows_to_frame(rows, column)


def _rows_to_frame(rows: list[dict[str, object]], column: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[column])
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index("date").sort_index()
