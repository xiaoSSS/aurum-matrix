"""SQLite helpers for future data snapshot caching."""

from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def open_sqlite_connection(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection and initialize the minimal snapshot table."""

    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            source TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)
    initialize_time_series_cache(connection)
    connection.commit()
    return connection


def initialize_time_series_cache(connection: sqlite3.Connection) -> None:
    """Create a generic time-series cache table shared by data clients."""

    connection.execute("""
        CREATE TABLE IF NOT EXISTS time_series_cache (
            source TEXT NOT NULL,
            series_id TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL,
            payload_json TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (source, series_id, date)
        )
        """)
    connection.commit()


def initialize_pricing_model_cache(connection: sqlite3.Connection) -> None:
    """Create cache table for fitted pricing-model coefficients."""

    connection.execute("""
        CREATE TABLE IF NOT EXISTS pricing_model_coefficients (
            frequency TEXT PRIMARY KEY,
            coefficients_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            fitted_at TEXT NOT NULL
        )
        """)
    connection.commit()


def write_pricing_model_coefficients(
    path: str | Path,
    *,
    frequency: str,
    coefficients: dict[str, float],
    metrics: dict[str, Any],
    train_start: str,
    train_end: str,
) -> None:
    """Persist fitted coefficients for later valuation and reporting use."""

    with sqlite3.connect(Path(path)) as connection:
        initialize_pricing_model_cache(connection)
        connection.execute(
            """
            INSERT OR REPLACE INTO pricing_model_coefficients (
                frequency, coefficients_json, metrics_json, train_start, train_end, fitted_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                frequency,
                json.dumps(coefficients, ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                train_start,
                train_end,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()


def read_pricing_model_coefficients(
    path: str | Path, *, frequency: str | None = None
) -> dict[str, dict[str, Any]]:
    """Read cached fitted coefficients, optionally for one frequency."""

    with sqlite3.connect(Path(path)) as connection:
        initialize_pricing_model_cache(connection)
        if frequency is None:
            rows = connection.execute(
                """
                SELECT frequency, coefficients_json, metrics_json, train_start, train_end, fitted_at
                FROM pricing_model_coefficients
                ORDER BY frequency
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT frequency, coefficients_json, metrics_json, train_start, train_end, fitted_at
                FROM pricing_model_coefficients
                WHERE frequency = ?
                """,
                (frequency,),
            ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for freq, coefficients_json, metrics_json, train_start, train_end, fitted_at in rows:
        try:
            coefficients = json.loads(coefficients_json)
        except json.JSONDecodeError:
            coefficients = {}
        try:
            metrics = json.loads(metrics_json)
        except json.JSONDecodeError:
            metrics = {}
        out[str(freq)] = {
            "coefficients": coefficients if isinstance(coefficients, dict) else {},
            "metrics": metrics if isinstance(metrics, dict) else {},
            "train_start": train_start,
            "train_end": train_end,
            "fitted_at": fitted_at,
        }
    return out


class TimeSeriesCache:
    """Small SQLite cache for normalized historical observations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with sqlite3.connect(self.path) as connection:
            initialize_time_series_cache(connection)

    def write_points(
        self,
        *,
        source: str,
        series_id: str,
        points: list[tuple[str, float | None, dict[str, Any] | None]],
    ) -> None:
        fetched_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as connection:
            initialize_time_series_cache(connection)
            connection.executemany(
                """
                INSERT OR REPLACE INTO time_series_cache (
                    source, series_id, date, value, payload_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        source,
                        series_id,
                        date,
                        value,
                        json.dumps(payload or {}, ensure_ascii=False),
                        fetched_at,
                    )
                    for date, value, payload in points
                ],
            )
            connection.commit()

    def read_points(
        self,
        *,
        source: str,
        series_id: str,
        max_age_hours: int | None = None,
    ) -> list[tuple[str, float | None, dict[str, Any]]]:
        cutoff: datetime | None = None
        if max_age_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        with sqlite3.connect(self.path) as connection:
            initialize_time_series_cache(connection)
            rows = connection.execute(
                """
                SELECT date, value, payload_json, fetched_at
                FROM time_series_cache
                WHERE source = ? AND series_id = ?
                ORDER BY date
                """,
                (source, series_id),
            ).fetchall()
        points: list[tuple[str, float | None, dict[str, Any]]] = []
        for date, value, payload_json, fetched_at in rows:
            if cutoff is not None:
                try:
                    fetched = datetime.fromisoformat(str(fetched_at))
                except ValueError:
                    continue
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                if fetched < cutoff:
                    continue
            try:
                payload = json.loads(payload_json or "{}")
            except json.JSONDecodeError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            points.append((str(date), value, payload))
        return points
