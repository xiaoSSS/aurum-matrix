"""Technical indicators used by the gold evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TechnicalIndicators(BaseModel):
    """Latest technical indicator values derived from price history."""

    ma_short: float = Field(description="Short moving average.")
    ma_long: float = Field(description="Long moving average.")
    macd: float
    macd_signal: float
    rsi: float = Field(ge=0, le=100)
    atr: float = Field(ge=0)


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate a simple moving average."""

    return series.rolling(window=window, min_periods=window).mean()


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """Calculate MACD line and signal line."""

    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""

    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(100)


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range from OHLC data."""

    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift()).abs()
    low_close = (frame["low"] - frame["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def calculate_technical_indicators(frame: pd.DataFrame) -> TechnicalIndicators | None:
    """Calculate latest MA, MACD, RSI and ATR values.

    Returns ``None`` if required columns or sufficient rows are missing.
    """

    required_columns = {"high", "low", "close"}
    if not required_columns.issubset(frame.columns) or len(frame) < 50:
        return None

    close = frame["close"].astype(float)
    macd_line, signal_line = macd(close)
    latest = TechnicalIndicators(
        ma_short=float(moving_average(close, 20).iloc[-1]),
        ma_long=float(moving_average(close, 50).iloc[-1]),
        macd=float(macd_line.iloc[-1]),
        macd_signal=float(signal_line.iloc[-1]),
        rsi=float(rsi(close).iloc[-1]),
        atr=float(atr(frame).iloc[-1]),
    )
    if any(np.isnan(value) for value in latest.model_dump().values()):
        return None
    return latest
