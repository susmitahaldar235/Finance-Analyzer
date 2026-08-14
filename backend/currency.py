"""
currency.py
Converts foreign-currency transactions into the user's base currency using
the Frankfurter API (https://frankfurter.dev) -- free, no API key required,
backed by European Central Bank reference rates.

Why this API and not a bank-linking service: it only ever sends a currency
pair and a date (e.g. "USD -> INR on 2026-05-12") -- never any transaction
amount, merchant name, or personal data. It's the one external network call
this project makes, and it's deliberately scoped to carry zero sensitive
information.

Rates are cached in SQLite (exchange_rates table) so:
  - the same currency/date pair is never re-fetched
  - the app still works (using last-known rates) if the API is briefly
    unreachable, as long as that pair was fetched before
"""

import requests
from database import get_conn

FRANKFURTER_URL = "https://api.frankfurter.dev/v1"
BASE_CURRENCY = "INR"   # change if you want a different home currency


def init_currency_table():
    """Called alongside database.init_db() at startup."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exchange_rates (
                date TEXT NOT NULL,
                from_currency TEXT NOT NULL,
                to_currency TEXT NOT NULL,
                rate REAL NOT NULL,
                PRIMARY KEY (date, from_currency, to_currency)
            )
        """)
        conn.commit()


def _get_cached_rate(date: str, from_currency: str, to_currency: str) -> float | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT rate FROM exchange_rates WHERE date = ? AND from_currency = ? AND to_currency = ?",
            (date, from_currency, to_currency)
        ).fetchone()
        return row["rate"] if row else None


def _cache_rate(date: str, from_currency: str, to_currency: str, rate: float):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO exchange_rates (date, from_currency, to_currency, rate)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date, from_currency, to_currency) DO UPDATE SET rate = excluded.rate
        """, (date, from_currency, to_currency, rate))
        conn.commit()


def get_rate(date: str, from_currency: str, to_currency: str = BASE_CURRENCY) -> float:
    """
    Returns the conversion rate (1 unit of from_currency = X units of to_currency)
    for the given historical date. Same-currency pairs always return 1.0 without
    a network call.
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    if from_currency == to_currency:
        return 1.0

    cached = _get_cached_rate(date, from_currency, to_currency)
    if cached is not None:
        return cached

    try:
        resp = requests.get(
            f"{FRANKFURTER_URL}/{date}",
            params={"base": from_currency, "symbols": to_currency},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = data["rates"][to_currency]
        _cache_rate(date, from_currency, to_currency, rate)
        return rate
    except Exception as e:
        print(f"[currency] Could not fetch rate for {from_currency}->{to_currency} on {date}: {e}")
        # Fall back to 1.0 (no conversion) rather than crashing the upload --
        # a wrong-but-visible amount is better than a failed upload. The
        # transaction is still flagged with its original currency so the
        # user can see something didn't convert.
        return 1.0


def convert_amount(amount: float, date: str, from_currency: str, to_currency: str = BASE_CURRENCY) -> float:
    rate = get_rate(date, from_currency, to_currency)
    return round(amount * rate, 2)
