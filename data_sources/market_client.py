"""Market data client abstraction and Alpha Vantage adapter."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from backend.config import AppConfig, get_settings

ALPHA_VANTAGE_API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"
ALPHA_VANTAGE_QUERY_URL = "https://www.alphavantage.co/query"
XAU_USD_SYMBOL = "XAU/USD"
DXY_SYMBOL = "DXY"
DEFAULT_DXY_PROVIDER_SYMBOL = "USDX"

MarketAsset = Literal["XAU/USD", "DXY"]
OutputSize = Literal["compact", "full"]


class MarketClientError(RuntimeError):
    """Base exception for explicit market client failures."""


class MarketApiKeyError(MarketClientError):
    """Raised when a provider API key is missing."""


class MarketRequestError(MarketClientError):
    """Raised when a market data request fails."""


class MarketResponseError(MarketClientError):
    """Raised when a market data response is invalid."""


class DailyBar(BaseModel):
    """Unified daily OHLCV bar.

    Alpha Vantage FX daily data does not include volume, so ``volume`` can be
    ``None`` while the field remains present in the unified schema.
    """

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = Field(default=None, ge=0)


class DailyMarketData(BaseModel):
    """Daily market data for one asset from one provider."""

    symbol: str
    provider: str
    bars: list[DailyBar]
    cached: bool = False


@dataclass(frozen=True)
class MarketDataRequest:
    """Cacheable market data request."""

    symbol: str
    function: str
    outputsize: OutputSize

    @property
    def cache_key(self) -> str:
        """Return a stable cache key for this market data request."""

        return "|".join([self.symbol, self.function, self.outputsize])


class MarketDataProvider(ABC):
    """Abstract interface for replaceable market data providers."""

    @abstractmethod
    def get_xau_usd_daily(
        self, *, outputsize: OutputSize = "compact"
    ) -> DailyMarketData:
        """Return XAU/USD daily bars."""

    @abstractmethod
    def get_dxy_daily(self, *, outputsize: OutputSize = "compact") -> DailyMarketData:
        """Return DXY daily bars."""

    @abstractmethod
    def get_daily(
        self, symbol: MarketAsset, *, outputsize: OutputSize = "compact"
    ) -> DailyMarketData:
        """Return daily bars for a supported market asset."""


class AlphaVantageMarketClient(MarketDataProvider):
    """Alpha Vantage implementation of the market data provider interface."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        timeout_seconds: float = 10.0,
        base_url: str = ALPHA_VANTAGE_QUERY_URL,
        dxy_symbol: str | None = None,
        settings: AppConfig | None = None,
    ) -> None:
        """Initialize Alpha Vantage client from explicit values or config."""

        active_settings = settings or get_settings()
        self.api_key = api_key or active_settings.alpha_vantage_api_key
        if not self.api_key:
            raise MarketApiKeyError(
                f"Missing required environment variable {ALPHA_VANTAGE_API_KEY_ENV}"
            )
        resolved_cache_path = (
            cache_path if cache_path is not None else active_settings.cache_db_path
        )
        self.cache_path = (
            Path(resolved_cache_path) if resolved_cache_path is not None else None
        )
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url
        self.dxy_symbol = dxy_symbol or active_settings.default_dxy_symbol
        if self.cache_path is not None:
            self._initialize_cache()

    def get_xau_usd_daily(
        self, *, outputsize: OutputSize = "compact"
    ) -> DailyMarketData:
        """Return XAU/USD proxy daily bars from GOLD_SILVER_HISTORY.

        Alpha Vantage does not support XAU/USD via FX_DAILY. We therefore use
        GOLD_SILVER_HISTORY(symbol=GOLD, interval=daily) and map the daily
        commodity price to unified OHLC fields.
        """

        request = MarketDataRequest(
            symbol=XAU_USD_SYMBOL,
            function="GOLD_SILVER_HISTORY",
            outputsize=outputsize,
        )
        payload, cached = self._get_payload(
            request,
            {
                "function": "GOLD_SILVER_HISTORY",
                "symbol": "GOLD",
                "interval": "daily",
            },
        )
        bars = _parse_alpha_vantage_gold_history(payload, symbol=XAU_USD_SYMBOL)
        return DailyMarketData(
            symbol=XAU_USD_SYMBOL,
            provider="alpha_vantage",
            bars=bars,
            cached=cached,
        )

    def get_dxy_daily(self, *, outputsize: OutputSize = "compact") -> DailyMarketData:
        """Return DXY daily bars from Alpha Vantage TIME_SERIES_DAILY."""

        provider_symbol = self.dxy_symbol
        request = MarketDataRequest(
            symbol=provider_symbol,
            function="TIME_SERIES_DAILY",
            outputsize=outputsize,
        )
        try:
            payload, cached = self._get_payload(
                request,
                {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": provider_symbol,
                    "outputsize": outputsize,
                },
            )
        except MarketResponseError:
            if provider_symbol.upper() == DEFAULT_DXY_PROVIDER_SYMBOL:
                raise
            fallback_symbol = DEFAULT_DXY_PROVIDER_SYMBOL
            fallback_request = MarketDataRequest(
                symbol=fallback_symbol,
                function="TIME_SERIES_DAILY",
                outputsize=outputsize,
            )
            payload, cached = self._get_payload(
                fallback_request,
                {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": fallback_symbol,
                    "outputsize": outputsize,
                },
            )
            provider_symbol = fallback_symbol
        bars = _parse_alpha_vantage_time_series(
            payload,
            series_key="Time Series (Daily)",
            symbol=provider_symbol,
            include_volume=True,
        )
        if len(bars) < 2 and provider_symbol.upper() != DEFAULT_DXY_PROVIDER_SYMBOL:
            fallback_symbol = DEFAULT_DXY_PROVIDER_SYMBOL
            fallback_request = MarketDataRequest(
                symbol=fallback_symbol,
                function="TIME_SERIES_DAILY",
                outputsize=outputsize,
            )
            payload, cached = self._get_payload(
                fallback_request,
                {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": fallback_symbol,
                    "outputsize": outputsize,
                },
            )
            bars = _parse_alpha_vantage_time_series(
                payload,
                series_key="Time Series (Daily)",
                symbol=fallback_symbol,
                include_volume=True,
            )
        return DailyMarketData(
            symbol=DXY_SYMBOL,
            provider="alpha_vantage",
            bars=bars,
            cached=cached,
        )

    def get_daily(
        self, symbol: MarketAsset, *, outputsize: OutputSize = "compact"
    ) -> DailyMarketData:
        """Return daily bars for a supported market asset."""

        if symbol == XAU_USD_SYMBOL:
            return self.get_xau_usd_daily(outputsize=outputsize)
        if symbol == DXY_SYMBOL:
            return self.get_dxy_daily(outputsize=outputsize)
        raise MarketResponseError(f"Unsupported market symbol: {symbol}")

    def _get_payload(
        self, request: MarketDataRequest, params_without_key: dict[str, str]
    ) -> tuple[dict[str, Any], bool]:
        """Return cached or newly requested provider payload."""

        cached_payload = self._read_cache(request)
        if cached_payload is not None:
            return cached_payload, True

        payload = self._request(params_without_key, request)
        self._write_cache(request, payload)
        return payload, False

    def _request(
        self, params_without_key: dict[str, str], request: MarketDataRequest
    ) -> dict[str, Any]:
        """Request Alpha Vantage and return decoded JSON with explicit errors."""

        params = {**params_without_key, "apikey": self.api_key}
        try:
            response = requests.get(
                self.base_url, params=params, timeout=self.timeout_seconds
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise MarketRequestError(
                f"Alpha Vantage request failed for symbol={request.symbol}: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise MarketResponseError(
                f"Alpha Vantage returned non-JSON response for symbol={request.symbol}"
            ) from exc

        if not isinstance(payload, dict):
            raise MarketResponseError(
                f"Alpha Vantage response must be an object for symbol={request.symbol}"
            )
        if "Error Message" in payload:
            raise MarketResponseError(
                f"Alpha Vantage error for symbol={request.symbol}: {payload['Error Message']}"
            )
        if "Note" in payload:
            raise MarketResponseError(
                f"Alpha Vantage rate limit for symbol={request.symbol}: {payload['Note']}"
            )
        if "Information" in payload:
            raise MarketResponseError(
                f"Alpha Vantage information response for symbol={request.symbol}: "
                f"{payload['Information']}"
            )
        return payload

    def _initialize_cache(self) -> None:
        """Create market data cache table if needed."""

        if self.cache_path is None:
            return
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    cache_key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    function TEXT NOT NULL,
                    outputsize TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                )
                """)
            connection.commit()

    def _read_cache(self, request: MarketDataRequest) -> dict[str, Any] | None:
        """Read cached market payload if present."""

        if self.cache_path is None:
            return None
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                "SELECT payload_json FROM market_data_cache WHERE cache_key = ?",
                (request.cache_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise MarketResponseError(
                f"Cached market payload is invalid for symbol={request.symbol}"
            ) from exc
        if not isinstance(payload, dict):
            raise MarketResponseError(
                f"Cached market payload must be an object for symbol={request.symbol}"
            )
        return payload

    def _write_cache(self, request: MarketDataRequest, payload: dict[str, Any]) -> None:
        """Write market payload to SQLite cache if enabled."""

        if self.cache_path is None:
            return
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO market_data_cache (
                    cache_key,
                    symbol,
                    function,
                    outputsize,
                    payload_json,
                    fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.cache_key,
                    request.symbol,
                    request.function,
                    request.outputsize,
                    json.dumps(payload),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()


def _parse_alpha_vantage_time_series(
    payload: dict[str, Any], *, series_key: str, symbol: str, include_volume: bool
) -> list[DailyBar]:
    """Parse Alpha Vantage time-series payload into unified daily bars."""

    raw_series = payload.get(series_key)
    if not isinstance(raw_series, dict):
        raise MarketResponseError(
            f"Alpha Vantage response missing {series_key!r} for symbol={symbol}"
        )

    bars: list[DailyBar] = []
    for date, raw_bar in sorted(raw_series.items()):
        if not isinstance(date, str) or not isinstance(raw_bar, dict):
            raise MarketResponseError(
                f"Invalid daily bar structure for symbol={symbol}"
            )
        bars.append(
            DailyBar(
                date=date,
                open=_parse_float_field(raw_bar, "1. open", symbol=symbol, date=date),
                high=_parse_float_field(raw_bar, "2. high", symbol=symbol, date=date),
                low=_parse_float_field(raw_bar, "3. low", symbol=symbol, date=date),
                close=_parse_float_field(raw_bar, "4. close", symbol=symbol, date=date),
                volume=(
                    _parse_float_field(raw_bar, "5. volume", symbol=symbol, date=date)
                    if include_volume
                    else None
                ),
            )
        )
    if not bars:
        raise MarketResponseError(
            f"Alpha Vantage returned no daily bars for symbol={symbol}"
        )
    return bars


def _parse_alpha_vantage_gold_history(
    payload: dict[str, Any], *, symbol: str
) -> list[DailyBar]:
    """Parse GOLD_SILVER_HISTORY payload into unified daily bars."""

    raw_series = payload.get("data")
    if not isinstance(raw_series, list):
        raise MarketResponseError(
            f"Alpha Vantage response missing 'data' for symbol={symbol}"
        )

    bars: list[DailyBar] = []
    for item in sorted(raw_series, key=lambda x: x.get("date", "")):
        if not isinstance(item, dict):
            raise MarketResponseError(
                f"Invalid daily bar structure for symbol={symbol}"
            )
        date = item.get("date")
        if not isinstance(date, str):
            raise MarketResponseError(
                f"Invalid daily bar date for symbol={symbol}: {date!r}"
            )
        raw_price = item.get("price")
        try:
            price = float(raw_price)
        except (TypeError, ValueError) as exc:
            raise MarketResponseError(
                f"Invalid 'price' value for symbol={symbol} date={date}: {raw_price!r}"
            ) from exc
        bars.append(
            DailyBar(
                date=date,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=None,
            )
        )
    if not bars:
        raise MarketResponseError(
            f"Alpha Vantage returned no daily bars for symbol={symbol}"
        )
    return bars


def _parse_float_field(
    raw_bar: dict[str, Any], field_name: str, *, symbol: str, date: str
) -> float:
    """Parse one Alpha Vantage numeric OHLCV field."""

    raw_value = raw_bar.get(field_name)
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise MarketResponseError(
            f"Invalid {field_name!r} value for symbol={symbol} date={date}: {raw_value!r}"
        ) from exc
