"""SQLite helpers for future data snapshot caching."""

from __future__ import annotations

import sqlite3
from pathlib import Path


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
    connection.commit()
    return connection
