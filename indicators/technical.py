"""Technical indicators used by the gold evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class IndicatorResult:
    """A single technical indicator with its latest value and full series."""

    latest_value: float
    series: pd.Series


@dataclass(frozen=True)
class MACDResult:
    """MACD indicator output with latest values and full output series."""

    latest_macd: float
    latest_signal: float
    latest_histogram: float
    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


class TechnicalIndicators(BaseModel):
    """Latest technical indicator values derived from price history."""

    ma_short: float = Field(description="Short moving average.")
    ma_long: float = Field(description="Long moving average.")
    macd: float
    macd_signal: float
    rsi: float = Field(ge=0, le=100)
    atr: float = Field(ge=0)


def _validate_series(series: pd.Series, *, name: str, min_length: int) -> pd.Series:
    """Return a numeric Series or raise ValueError for invalid input."""

    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    if len(series) < min_length:
        raise ValueError(f"{name} length must be at least {min_length}")

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any():
        raise ValueError(f"{name} must not contain NaN or non-numeric values")
    return numeric


def _validate_window(window: int, *, name: str = "window") -> None:
    """Validate a positive rolling/EMA window."""

    if window <= 0:
        raise ValueError(f"{name} must be greater than 0")


def simple_moving_average(series: pd.Series, window: int) -> IndicatorResult:
    """Calculate simple moving average and return latest value plus full series."""

    _validate_window(window)
    numeric = _validate_series(series, name="series", min_length=window)
    result = numeric.rolling(window=window, min_periods=window).mean()
    latest_value = result.iloc[-1]
    if pd.isna(latest_value):
        raise ValueError("simple moving average latest value is NaN")
    return IndicatorResult(latest_value=float(latest_value), series=result)


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Backward-compatible helper returning only the SMA series."""

    return simple_moving_average(series, window).series


def ema(series: pd.Series, window: int) -> IndicatorResult:
    """Calculate exponential moving average and return latest value plus full series."""

    _validate_window(window)
    numeric = _validate_series(series, name="series", min_length=window)
    result = numeric.ewm(span=window, adjust=False, min_periods=window).mean()
    latest_value = result.iloc[-1]
    if pd.isna(latest_value):
        raise ValueError("EMA latest value is NaN")
    return IndicatorResult(latest_value=float(latest_value), series=result)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> MACDResult:
    """Calculate MACD line, signal line and histogram.

    The returned object includes both the latest values and the full output
    series for each MACD component.
    """

    _validate_window(fast, name="fast")
    _validate_window(slow, name="slow")
    _validate_window(signal, name="signal")
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")

    min_length = slow + signal - 1
    numeric = _validate_series(close, name="close", min_length=min_length)
    fast_ema = ema(numeric, fast).series
    slow_ema = ema(numeric, slow).series
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line

    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_histogram = histogram.iloc[-1]
    if pd.isna(latest_macd) or pd.isna(latest_signal) or pd.isna(latest_histogram):
        raise ValueError("MACD latest values must not be NaN")

    return MACDResult(
        latest_macd=float(latest_macd),
        latest_signal=float(latest_signal),
        latest_histogram=float(latest_histogram),
        macd=macd_line,
        signal=signal_line,
        histogram=histogram,
    )


def rsi(close: pd.Series, window: int = 14) -> IndicatorResult:
    """Calculate Relative Strength Index and return latest value plus full series."""

    _validate_window(window)
    numeric = _validate_series(close, name="close", min_length=window + 1)
    delta = numeric.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()

    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((loss == 0) & (gain > 0), 100.0)
    result = result.mask((loss == 0) & (gain == 0), 50.0)

    latest_value = result.iloc[-1]
    if pd.isna(latest_value):
        raise ValueError("RSI latest value is NaN")
    return IndicatorResult(latest_value=float(latest_value), series=result)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> IndicatorResult:
    """Calculate Average True Range and return latest value plus full series."""

    _validate_window(window)
    high_numeric = _validate_series(high, name="high", min_length=window)
    low_numeric = _validate_series(low, name="low", min_length=window)
    close_numeric = _validate_series(close, name="close", min_length=window)
    if not (len(high_numeric) == len(low_numeric) == len(close_numeric)):
        raise ValueError("high, low and close must have the same length")
    if not (
        high_numeric.index.equals(low_numeric.index)
        and high_numeric.index.equals(close_numeric.index)
    ):
        raise ValueError("high, low and close must have matching indexes")
    if (high_numeric < low_numeric).any():
        raise ValueError("high values must be greater than or equal to low values")

    high_low = high_numeric - low_numeric
    high_close = (high_numeric - close_numeric.shift()).abs()
    low_close = (low_numeric - close_numeric.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result = true_range.rolling(window=window, min_periods=window).mean()

    latest_value = result.iloc[-1]
    if pd.isna(latest_value):
        raise ValueError("ATR latest value is NaN")
    return IndicatorResult(latest_value=float(latest_value), series=result)


def calculate_technical_indicators(frame: pd.DataFrame) -> TechnicalIndicators | None:
    """Calculate latest MA, MACD, RSI and ATR values.

    Returns ``None`` if required columns or sufficient rows are missing.
    """

    required_columns = {"high", "low", "close"}
    if not required_columns.issubset(frame.columns) or len(frame) < 50:
        return None

    try:
        close = frame["close"].astype(float)
        macd_result = macd(close)
        latest = TechnicalIndicators(
            ma_short=simple_moving_average(close, 20).latest_value,
            ma_long=simple_moving_average(close, 50).latest_value,
            macd=macd_result.latest_macd,
            macd_signal=macd_result.latest_signal,
            rsi=rsi(close).latest_value,
            atr=atr(frame["high"], frame["low"], close).latest_value,
        )
    except (TypeError, ValueError):
        return None

    if any(np.isnan(value) for value in latest.model_dump().values()):
        return None
    return latest
