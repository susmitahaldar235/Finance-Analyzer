"""
csv_parser.py
Parses a bank/credit-card CSV export into a clean list of transaction dicts.

Scope decision (intentional, documented in README): this project supports
CSV only, one flexible-but-single schema shape. PDF parsing and multi-bank
format normalization are noted as future work -- trying to handle every
bank's PDF layout is the single biggest time-sink in a project like this,
and CSV export is available from virtually every bank/card provider.

Expected input columns (case-insensitive, order-independent):
    date, description (or merchant/narration), amount
Optional:
    type (debit/credit) -- if present, credits are treated as positive income
                            and debits as spend; if absent we assume all
                            amounts in the file are already signed correctly
                            (positive = spend, negative = credit/refund).
"""

import pandas as pd
from datetime import datetime
from currency import BASE_CURRENCY

# Common column name variants seen across bank exports, mapped to our canonical names
COLUMN_ALIASES = {
    "date": ["date", "transaction date", "txn date", "value date"],
    "merchant": ["description", "merchant", "narration", "particulars", "details"],
    "amount": ["amount", "debit", "withdrawal amt", "transaction amount"],
    "type": ["type", "transaction type", "dr/cr"],
    "currency": ["currency", "ccy", "currency code"],
}


def _find_column(columns_lower: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in columns_lower:
            return alias
    return None


def _parse_date(raw: str) -> str:
    """Try a handful of common date formats; return ISO 'YYYY-MM-DD'.
    Raises ValueError if nothing matches -- caller should skip/flag that row."""
    raw = str(raw).strip()
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%b %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {raw}")


def parse_csv(file_path: str, source_filename: str = "") -> tuple[list[dict], list[str]]:
    """
    Returns (transactions, warnings).
    transactions: list of dicts with keys: date, month, merchant, amount, source_file
                  (category/merchant_normalized are added later by categorizer.py)
    warnings: human-readable strings about skipped/malformed rows, surfaced to the UI
              instead of failing the whole upload.
    """
    warnings = []
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    columns_lower = [c.lower() for c in df.columns]

    date_col = _find_column(columns_lower, COLUMN_ALIASES["date"])
    merchant_col = _find_column(columns_lower, COLUMN_ALIASES["merchant"])
    amount_col = _find_column(columns_lower, COLUMN_ALIASES["amount"])

    if not (date_col and merchant_col and amount_col):
        missing = []
        if not date_col: missing.append("date")
        if not merchant_col: missing.append("merchant/description")
        if not amount_col: missing.append("amount")
        raise ValueError(
            f"Could not find required column(s): {', '.join(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    # map lowercased alias back to the real column name in the dataframe
    real_date_col = df.columns[columns_lower.index(date_col)]
    real_merchant_col = df.columns[columns_lower.index(merchant_col)]
    real_amount_col = df.columns[columns_lower.index(amount_col)]

    # Currency column is optional -- most single-country bank exports won't have one.
    # If absent, every row is assumed to already be in BASE_CURRENCY.
    currency_col = _find_column(columns_lower, COLUMN_ALIASES["currency"])
    real_currency_col = df.columns[columns_lower.index(currency_col)] if currency_col else None

    transactions = []
    for i, row in df.iterrows():
        try:
            date_iso = _parse_date(row[real_date_col])
        except ValueError as e:
            warnings.append(f"Row {i+2}: skipped -- {e}")
            continue

        merchant = str(row[real_merchant_col]).strip()
        if not merchant or merchant.lower() == "nan":
            warnings.append(f"Row {i+2}: skipped -- empty merchant/description")
            continue

        try:
            amount = float(str(row[real_amount_col]).replace(",", "").replace("₹", "").replace("$", "").strip())
        except (ValueError, TypeError):
            warnings.append(f"Row {i+2}: skipped -- unparseable amount '{row[real_amount_col]}'")
            continue

        row_currency = BASE_CURRENCY
        if real_currency_col is not None:
            raw_ccy = str(row[real_currency_col]).strip().upper()
            if raw_ccy and raw_ccy != "NAN":
                row_currency = raw_ccy

        transactions.append({
            "date": date_iso,
            "month": date_iso[:7],   # 'YYYY-MM'
            "merchant": merchant,
            "original_amount": amount,        # amount as it appeared in the statement
            "original_currency": row_currency, # currency it was recorded in
            "source_file": source_filename,
        })

    if not transactions:
        warnings.append("No valid transactions found in this file.")

    return transactions, warnings
