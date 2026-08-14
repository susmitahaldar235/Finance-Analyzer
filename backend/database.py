"""
database.py
SQLite storage for the Personal Finance Analyzer.

Two tables:
1. transactions            -> every parsed transaction, tagged with category + month
2. merchant_category_cache -> merchant -> category, so we never re-ask the LLM
                               for a merchant we've already categorized before.

SQLite (not Postgres/MySQL) is a deliberate choice: this project's whole pitch
is "local-first, nothing leaves your machine," and a single-file DB reinforces
that instead of adding infra that undermines it.
"""

import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

# Locally this defaults to a file next to this script. Inside Docker,
# docker-compose sets DB_PATH=/app/data/finance.db so the file lives on
# a mounted volume and survives container rebuilds.
DB_PATH = Path(os.environ.get("DB_PATH", Path(__file__).parent / "finance.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db():
    """Create tables if they don't exist. Safe to call every startup."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                month TEXT NOT NULL,                -- 'YYYY-MM', precomputed for fast grouping
                merchant TEXT NOT NULL,
                merchant_normalized TEXT NOT NULL,  -- lowercased/stripped for matching
                amount REAL NOT NULL,               -- converted amount, in BASE_CURRENCY -- all stats/charts use this
                original_amount REAL NOT NULL,      -- amount as it appeared in the statement, pre-conversion
                original_currency TEXT NOT NULL,    -- currency it was recorded in (e.g. 'USD')
                category TEXT NOT NULL,
                source_file TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS merchant_category_cache (
                merchant_normalized TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                source TEXT DEFAULT 'rule',         -- 'rule' or 'llm' -- useful for demo/debugging
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_month ON transactions(month)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transactions(merchant_normalized)")
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_transactions(rows: list[dict]):
    with get_conn() as conn:
        conn.executemany("""
            INSERT INTO transactions
                (date, month, merchant, merchant_normalized, amount, original_amount, original_currency, category, source_file)
            VALUES
                (:date, :month, :merchant, :merchant_normalized, :amount, :original_amount, :original_currency, :category, :source_file)
        """, rows)
        conn.commit()


def get_cached_category(merchant_normalized: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT category FROM merchant_category_cache WHERE merchant_normalized = ?",
            (merchant_normalized,)
        ).fetchone()
        return row["category"] if row else None


def cache_category(merchant_normalized: str, category: str, source: str = "rule"):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO merchant_category_cache (merchant_normalized, category, source, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(merchant_normalized) DO UPDATE SET
                category = excluded.category,
                source = excluded.source,
                updated_at = datetime('now')
        """, (merchant_normalized, category, source))
        conn.commit()


def get_all_transactions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM transactions ORDER BY date").fetchall()
        return [dict(r) for r in rows]


def get_transactions_by_month(month: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM transactions WHERE month = ? ORDER BY date", (month,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_distinct_months() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT month FROM transactions ORDER BY month").fetchall()
        return [r["month"] for r in rows]


def clear_all_data():
    """Reset button used by the UI / for re-testing."""
    with get_conn() as conn:
        conn.execute("DELETE FROM transactions")
        conn.execute("DELETE FROM merchant_category_cache")
        conn.commit()
