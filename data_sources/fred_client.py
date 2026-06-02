"""FRED API client with explicit errors and optional SQLite caching."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field

from backend.config import AppConfig, get_settings

FRED_API_KEY_ENV = "FRED_API_KEY"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
TEN_YEAR_REAL_RATE_SERIES_ID = "DFII10"
TEN_YEAR_TREASURY_YIELD_SERIES_ID = "DGS10"
TEN_YEAR_BREAKEVEN_INFLATION_SERIES_ID = "T10YIE"
US_TOTAL_PUBLIC_DEBT_SERIES_ID = "GFDEBTN"
CPI_INDEX_SERIES_ID = "CPIAUCSL"


class FredClientError(RuntimeError):
    """Base exception for explicit FRED client failures."""


class FredApiKeyError(FredClientError):
    """Raised when the FRED API key is missing."""


class FredRequestError(FredClientError):
    """Raised when an HTTP request to FRED fails."""


class FredResponseError(FredClientError):
    """Raised when the FRED response cannot be parsed as expected."""


class FredObservation(BaseModel):
    """Single FRED observation."""

    date: str
    value: float | None = Field(
        description="Observation value. FRED missing value marker '.' is represented as None."
    )


class FredSeriesData(BaseModel):
    """Historical data returned for one FRED series."""

    series_id: str
    observations: list[FredObservation]
    source: str = "FRED"
    cached: bool = False


@dataclass(frozen=True)
class FredSeriesRequest:
    """Cacheable FRED series request parameters."""

    series_id: str
    observation_start: str | None = None
    observation_end: str | None = None

    @property
    def cache_key(self) -> str:
        """Return a stable cache key for this request."""

        return "|".join(
            [
                self.series_id,
                self.observation_start or "",
                self.observation_end or "",
            ]
        )


class FredClient:
    """Small FRED client for historical macro series used by Aurum Matrix."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        timeout_seconds: float = 10.0,
        base_url: str = FRED_OBSERVATIONS_URL,
        settings: AppConfig | None = None,
    ) -> None:
        """Initialize the client from explicit values or centralized config."""

        active_settings = settings or get_settings()
        self.api_key = api_key or active_settings.fred_api_key
        if not self.api_key:
            raise FredApiKeyError(
                f"Missing required environment variable {FRED_API_KEY_ENV}"
            )

        resolved_cache_path = (
            cache_path if cache_path is not None else active_settings.cache_db_path
        )
        self.cache_path = (
            Path(resolved_cache_path) if resolved_cache_path is not None else None
        )
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url
        if self.cache_path is not None:
            self._initialize_cache()

    def get_series_history(
        self,
        series_id: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> FredSeriesData:
        """Fetch historical observations for a FRED series, using SQLite cache if enabled."""

        request = FredSeriesRequest(
            series_id=series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        cached_payload = self._read_cache(request)
        if cached_payload is not None:
            return self._parse_series_payload(series_id, cached_payload, cached=True)

        payload = self._request_series(request)
        self._write_cache(request, payload)
        return self._parse_series_payload(series_id, payload, cached=False)

    def get_10y_real_rate(
        self,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> FredSeriesData:
        """Fetch the 10-year real interest rate series (DFII10)."""

        return self.get_series_history(
            TEN_YEAR_REAL_RATE_SERIES_ID,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    def get_10y_treasury_yield(
        self,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> FredSeriesData:
        """Fetch the 10-year Treasury constant maturity yield series (DGS10)."""

        return self.get_series_history(
            TEN_YEAR_TREASURY_YIELD_SERIES_ID,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    def get_inflation_expectations(
        self,
        *,
        series_id: str = TEN_YEAR_BREAKEVEN_INFLATION_SERIES_ID,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> FredSeriesData:
        """Fetch an inflation-expectations related series, defaulting to T10YIE."""

        return self.get_series_history(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    def get_us_total_public_debt(
        self,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> FredSeriesData:
        """Fetch total public debt (GFDEBTN)."""

        return self.get_series_history(
            US_TOTAL_PUBLIC_DEBT_SERIES_ID,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    def get_cpi_index(
        self,
        *,
        series_id: str = CPI_INDEX_SERIES_ID,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> FredSeriesData:
        """Fetch inflation index observations, defaulting to CPIAUCSL."""

        return self.get_series_history(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )

    def _request_series(self, request: FredSeriesRequest) -> dict[str, Any]:
        """Request a FRED series and return decoded JSON, raising clear errors."""

        params: dict[str, str] = {
            "series_id": request.series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if request.observation_start is not None:
            params["observation_start"] = request.observation_start
        if request.observation_end is not None:
            params["observation_end"] = request.observation_end

        try:
            response = requests.get(
                self.base_url, params=params, timeout=self.timeout_seconds
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FredRequestError(
                f"FRED request failed for series_id={request.series_id}: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise FredResponseError(
                f"FRED returned non-JSON response for series_id={request.series_id}"
            ) from exc

        if "error_code" in payload:
            message = payload.get("error_message", "unknown FRED API error")
            raise FredResponseError(
                f"FRED API error for series_id={request.series_id}: {message}"
            )
        if "observations" not in payload or not isinstance(
            payload["observations"], list
        ):
            raise FredResponseError(
                f"FRED response missing observations for series_id={request.series_id}"
            )
        return payload

    def _initialize_cache(self) -> None:
        """Create the FRED cache table if needed."""

        if self.cache_path is None:
            return
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS fred_series_cache (
                    cache_key TEXT PRIMARY KEY,
                    series_id TEXT NOT NULL,
                    observation_start TEXT,
                    observation_end TEXT,
                    payload_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                )
                """)
            connection.commit()

    def _read_cache(self, request: FredSeriesRequest) -> dict[str, Any] | None:
        """Read a cached FRED response payload when available."""

        if self.cache_path is None:
            return None
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                "SELECT payload_json FROM fred_series_cache WHERE cache_key = ?",
                (request.cache_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise FredResponseError(
                f"Cached FRED payload is invalid for series_id={request.series_id}"
            ) from exc
        if not isinstance(payload, dict):
            raise FredResponseError(
                f"Cached FRED payload must be an object for series_id={request.series_id}"
            )
        return payload

    def _write_cache(self, request: FredSeriesRequest, payload: dict[str, Any]) -> None:
        """Write a FRED response payload to SQLite cache when caching is enabled."""

        if self.cache_path is None:
            return
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO fred_series_cache (
                    cache_key,
                    series_id,
                    observation_start,
                    observation_end,
                    payload_json,
                    fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.cache_key,
                    request.series_id,
                    request.observation_start,
                    request.observation_end,
                    json.dumps(payload),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()

    @staticmethod
    def _parse_series_payload(
        series_id: str, payload: dict[str, Any], *, cached: bool
    ) -> FredSeriesData:
        """Parse a FRED payload into typed observations."""

        observations_payload = payload.get("observations")
        if not isinstance(observations_payload, list):
            raise FredResponseError(
                f"FRED response missing observations for series_id={series_id}"
            )

        observations: list[FredObservation] = []
        for item in observations_payload:
            if not isinstance(item, dict):
                raise FredResponseError(
                    f"FRED observation must be an object for series_id={series_id}"
                )
            date = item.get("date")
            raw_value = item.get("value")
            if not isinstance(date, str) or raw_value is None:
                raise FredResponseError(
                    f"FRED observation missing date/value for series_id={series_id}"
                )
            observations.append(
                FredObservation(date=date, value=_parse_observation_value(raw_value))
            )
        return FredSeriesData(
            series_id=series_id,
            observations=observations,
            cached=cached,
        )


def _parse_observation_value(raw_value: Any) -> float | None:
    """Parse a FRED observation value, preserving FRED missing marker as None."""

    if raw_value == ".":
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise FredResponseError(
            f"Invalid FRED observation value: {raw_value!r}"
        ) from exc
