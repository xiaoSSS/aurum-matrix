"""Enhanced factor data clients for ETF, CFTC and central-bank gold flows."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel

from data_sources.sqlite_cache import TimeSeriesCache
from scoring.factor_score import CftcPositionState


class EnhancedObservation(BaseModel):
    date: str
    value: float | None = None
    payload: dict[str, Any] = {}


class EnhancedSeriesData(BaseModel):
    source: str
    series_id: str
    observations: list[EnhancedObservation]
    cached: bool = False
    warnings: list[str] = []


class EnhancedDataClient:
    """Generic API/file client for normalized enhanced-factor time series."""

    def __init__(
        self,
        *,
        source: str,
        series_id: str,
        url: str | None = None,
        cache_path: str | Path | None = None,
        cache_ttl_hours: int | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.source = source
        self.series_id = series_id
        self.url = url
        self.cache = TimeSeriesCache(cache_path) if cache_path is not None else None
        self.cache_ttl_hours = cache_ttl_hours
        self.timeout_seconds = timeout_seconds

    def get_series(self) -> EnhancedSeriesData:
        cached = self._read_cache()
        if cached:
            return EnhancedSeriesData(
                source=self.source,
                series_id=self.series_id,
                observations=cached,
                cached=True,
            )
        if not self.url:
            return EnhancedSeriesData(
                source=self.source,
                series_id=self.series_id,
                observations=[],
                warnings=[f"{self.series_id} source URL is not configured"],
            )
        try:
            text = self._read_text()
            observations = _parse_observations(text)
        except Exception as exc:
            fallback = self._read_cache(ignore_ttl=True)
            return EnhancedSeriesData(
                source=self.source,
                series_id=self.series_id,
                observations=fallback,
                cached=bool(fallback),
                warnings=[f"{self.series_id} fetch failed: {exc}"],
            )
        if self.cache is not None:
            self.cache.write_points(
                source=self.source,
                series_id=self.series_id,
                points=[(o.date, o.value, o.payload) for o in observations],
            )
        return EnhancedSeriesData(
            source=self.source,
            series_id=self.series_id,
            observations=observations,
        )

    def _read_text(self) -> str:
        if self.url and self.url.startswith(("http://", "https://")):
            response = requests.get(self.url, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.text
        return Path(str(self.url)).read_text(encoding="utf-8")

    def _read_cache(self, *, ignore_ttl: bool = False) -> list[EnhancedObservation]:
        if self.cache is None:
            return []
        rows = self.cache.read_points(
            source=self.source,
            series_id=self.series_id,
            max_age_hours=None if ignore_ttl else self.cache_ttl_hours,
        )
        return [
            EnhancedObservation(date=date, value=value, payload=payload)
            for date, value, payload in rows
        ]


def get_latest_value(series: EnhancedSeriesData) -> float | None:
    for obs in reversed(sorted(series.observations, key=lambda x: x.date)):
        if obs.value is not None:
            return obs.value
    return None


def trend_from_observations(series: EnhancedSeriesData) -> str | None:
    values = [
        obs.value
        for obs in sorted(series.observations, key=lambda x: x.date)
        if obs.value is not None
    ]
    if len(values) < 2:
        return None
    if values[-1] > values[-2]:
        return "up"
    if values[-1] < values[-2]:
        return "down"
    return "flat"


def cftc_position_state(series: EnhancedSeriesData) -> CftcPositionState | None:
    values = [
        obs.value
        for obs in sorted(series.observations, key=lambda x: x.date)
        if obs.value is not None
    ]
    if not values:
        return None
    latest = values[-1]
    previous = values[-2] if len(values) >= 2 else latest
    if latest < 0:
        return "net_short"
    if latest > 0 and latest > previous:
        return "net_long_increasing"
    if latest > 0 and latest > max(values) * 0.9:
        return "crowded_long"
    return "neutral"


def _parse_observations(text: str) -> list[EnhancedObservation]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("{") or stripped.startswith("["):
        return _parse_json_observations(stripped)
    return _parse_csv_observations(stripped)


def _parse_json_observations(text: str) -> list[EnhancedObservation]:
    import json

    payload = json.loads(text)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    observations: list[EnhancedObservation] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        date = row.get("date") or row.get("period") or row.get("report_date_as_yyyy_mm_dd")
        value = (
            row.get("value")
            or row.get("net_position")
            or row.get("flow_tonnes")
            or row.get("tonnes")
            or row.get("net_purchase_tonnes")
        )
        parsed = _parse_float(value)
        if isinstance(date, str):
            observations.append(EnhancedObservation(date=date, value=parsed, payload=row))
    return sorted(observations, key=lambda x: x.date)


def _parse_csv_observations(text: str) -> list[EnhancedObservation]:
    reader = csv.DictReader(StringIO(text))
    observations: list[EnhancedObservation] = []
    for row in reader:
        lowered = {k.lower().strip(): v for k, v in row.items() if k is not None}
        date = (
            lowered.get("date")
            or lowered.get("period")
            or lowered.get("report_date_as_yyyy_mm_dd")
        )
        value = (
            lowered.get("value")
            or lowered.get("net_position")
            or lowered.get("flow_tonnes")
            or lowered.get("tonnes")
            or lowered.get("net_purchase_tonnes")
        )
        if isinstance(date, str):
            observations.append(
                EnhancedObservation(date=date, value=_parse_float(value), payload=row)
            )
    return sorted(observations, key=lambda x: x.date)


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None
