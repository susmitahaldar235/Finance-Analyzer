"""
stats_engine.py
Pure computation over stored transactions -- no LLM calls here at all.

Rationale: anomaly/trend detection is fundamentally a statistics problem.
Using an LLM to "notice" spending patterns is slower, less reliable, and
harder to reason about than just computing the numbers directly. The LLM's
job (elsewhere) is to *explain* these numbers in plain English, not compute them.
"""

from collections import defaultdict
from statistics import mean, stdev
from database import get_all_transactions, get_transactions_by_month, get_distinct_months


def category_totals(month: str) -> dict[str, float]:
    """Total spend per category for a given month. Income excluded from 'spend'."""
    txns = get_transactions_by_month(month)
    totals = defaultdict(float)
    for t in txns:
        if t["category"] != "Income":
            totals[t["category"]] += t["amount"]
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def month_over_month(current_month: str) -> dict:
    """
    Compare current_month's category spend against the previous available month.
    Returns per-category absolute + percentage change, plus overall totals.
    """
    months = get_distinct_months()
    if current_month not in months:
        return {"error": f"No data for month {current_month}"}

    idx = months.index(current_month)
    if idx == 0:
        return {"note": "No prior month available for comparison.", "current": category_totals(current_month)}

    prev_month = months[idx - 1]
    current = category_totals(current_month)
    previous = category_totals(prev_month)

    all_categories = set(current) | set(previous)
    comparison = {}
    for cat in all_categories:
        cur_amt = current.get(cat, 0.0)
        prev_amt = previous.get(cat, 0.0)
        pct_change = ((cur_amt - prev_amt) / prev_amt * 100) if prev_amt else (100.0 if cur_amt else 0.0)
        comparison[cat] = {
            "current": round(cur_amt, 2),
            "previous": round(prev_amt, 2),
            "change_pct": round(pct_change, 1),
        }

    return {
        "current_month": current_month,
        "previous_month": prev_month,
        "total_current": round(sum(current.values()), 2),
        "total_previous": round(sum(previous.values()), 2),
        "by_category": dict(sorted(comparison.items(), key=lambda kv: -kv[1]["current"])),
    }


def detect_recurring_charges(min_occurrences: int = 2, amount_tolerance_pct: float = 15.0) -> list[dict]:
    """
    Group all transactions by merchant. Flag a merchant as 'recurring' if it
    appears in >= min_occurrences distinct months with amounts within
    amount_tolerance_pct of each other -- classic subscription-detection heuristic,
    no LLM needed.
    """
    txns = get_all_transactions()
    by_merchant = defaultdict(list)
    for t in txns:
        by_merchant[t["merchant_normalized"]].append(t)

    recurring = []
    for merchant, occurrences in by_merchant.items():
        months_seen = set(o["month"] for o in occurrences)
        if len(months_seen) < min_occurrences:
            continue

        amounts = [o["amount"] for o in occurrences]
        avg_amount = mean(amounts)
        if avg_amount == 0:
            continue
        # check all amounts are within tolerance of the average (consistent recurring charge)
        consistent = all(abs(a - avg_amount) / avg_amount * 100 <= amount_tolerance_pct for a in amounts)
        if consistent:
            recurring.append({
                "merchant": occurrences[0]["merchant"],
                "category": occurrences[0]["category"],
                "months_seen": sorted(months_seen),
                "occurrence_count": len(occurrences),
                "average_amount": round(avg_amount, 2),
            })

    return sorted(recurring, key=lambda r: -r["average_amount"])


def detect_anomalies(month: str, z_threshold: float = 1.5) -> list[dict]:
    """
    Flag categories in `month` whose spend deviates significantly from that
    category's historical average across all prior months (simple z-score-style
    check, not a full statistical model -- good enough for a personal finance tool).
    """
    months = get_distinct_months()
    if month not in months or months.index(month) == 0:
        return []

    history_months = months[:months.index(month)]
    history_totals = defaultdict(list)
    for m in history_months:
        for cat, amt in category_totals(m).items():
            history_totals[cat].append(amt)

    current = category_totals(month)
    anomalies = []
    for cat, amt in current.items():
        history = history_totals.get(cat, [])
        if len(history) < 2:
            continue  # not enough history to judge "unusual"
        avg = mean(history)
        try:
            sd = stdev(history)
        except Exception:
            sd = 0
        if sd == 0:
            continue
        z = (amt - avg) / sd
        if abs(z) >= z_threshold:
            anomalies.append({
                "category": cat,
                "current_amount": round(amt, 2),
                "historical_average": round(avg, 2),
                "z_score": round(z, 2),
                "direction": "above" if z > 0 else "below",
            })

    return sorted(anomalies, key=lambda a: -abs(a["z_score"]))


def overview_summary() -> dict:
    """One-shot bundle used by the dashboard on load."""
    months = get_distinct_months()
    if not months:
        return {"months": [], "message": "No data uploaded yet."}
    latest = months[-1]
    return {
        "months": months,
        "latest_month": latest,
        "category_totals": category_totals(latest),
        "month_over_month": month_over_month(latest),
        "recurring_charges": detect_recurring_charges(),
        "anomalies": detect_anomalies(latest),
    }
