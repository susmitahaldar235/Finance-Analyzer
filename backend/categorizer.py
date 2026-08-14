"""
categorizer.py
Two-tier transaction categorization:

    Tier 1: rule-based keyword matching  -> fast, free, deterministic
    Tier 2: LLM fallback (batched)       -> only for merchants the rules miss

Why this design (say this in interviews):
  - Calling an LLM per-transaction is slow and wasteful. Most merchants follow
    predictable naming patterns, so a keyword lookup handles the large majority
    instantly and for free.
  - The LLM is invoked only for the unmatched subset, and always in a single
    batched call (one round trip for N merchants), not one call per transaction --
    this is the key latency-saving decision that keeps this usable on CPU-only
    local inference.
  - Every LLM decision is cached in merchant_category_cache (see database.py)
    so the same merchant is never sent to the LLM twice, across uploads.
"""

import re
import json
from ollama_client import call_ollama
from database import get_cached_category, cache_category

CATEGORIES = [
    "Groceries", "Food Delivery", "Shopping", "Subscriptions",
    "Transport", "Utilities", "Rent", "Entertainment",
    "Healthcare", "Travel", "Fees & Charges", "Income", "Other"
]

# keyword -> category, matched against a normalized (lowercased, ID-stripped) merchant string.
# Small and readable on purpose -- extend as you see more real statement data.
RULES = {
    "Groceries": ["bigbasket", "blinkit", "zepto", "grofers", "dmart", "grocery", "reliance fresh", "more supermarket"],
    "Food Delivery": ["swiggy", "zomato", "ubereats", "dominos", "pizza"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa"],
    "Subscriptions": ["netflix", "spotify", "prime video", "hotstar", "youtube premium", "audible", "icloud", "google one"],
    "Transport": ["uber", "ola", "rapido", "irctc", "petrol", "fuel", "metro"],
    "Utilities": ["electricity", "airtel", "jio", "vodafone", "broadband", "wifi", "gas board", "water bill"],
    "Rent": ["rent", "landlord", "housing.com"],
    "Entertainment": ["bookmyshow", "pvr", "inox", "steam", "playstation"],
    "Healthcare": ["pharmacy", "apollo", "medplus", "hospital", "clinic", "1mg", "practo"],
    "Travel": ["makemytrip", "goibibo", "airbnb", "indigo", "vistara", "airline"],
    "Fees & Charges": ["annual fee", "late fee", "interest charge", "gst", "penalty", "atm withdrawal charge"],
    "Income": ["salary", "credited", "refund", "cashback", "reversal"],
}


def normalize_merchant(raw: str) -> str:
    """Strip transaction-ID noise so 'SWIGGY*BLR8821' and 'SWIGGY*DEL1123'
    both normalize to the same cache key."""
    s = raw.lower().strip()
    s = re.sub(r"[*#].*$", "", s)     # drop trailing IDs after * or #
    s = re.sub(r"\d{4,}", "", s)      # drop long digit runs (txn refs, card suffixes)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rule_categorize(merchant_normalized: str) -> str | None:
    for category, keywords in RULES.items():
        for kw in keywords:
            if kw in merchant_normalized:
                return category
    return None


def categorize_transactions(transactions: list[dict]) -> list[dict]:
    """
    Mutates each transaction dict to add 'merchant_normalized' and 'category'.
    Resolution order: persistent cache -> keyword rules -> batched LLM fallback.
    """
    uncategorized_indices = []
    uncategorized_merchants = []

    for i, txn in enumerate(transactions):
        norm = normalize_merchant(txn["merchant"])
        txn["merchant_normalized"] = norm

        cached = get_cached_category(norm)
        if cached:
            txn["category"] = cached
            continue

        rule_match = rule_categorize(norm)
        if rule_match:
            txn["category"] = rule_match
            cache_category(norm, rule_match, source="rule")
            continue

        txn["category"] = None
        uncategorized_indices.append(i)
        uncategorized_merchants.append(norm)

    if uncategorized_merchants:
        llm_results = llm_categorize_batch(uncategorized_merchants)
        for idx, merchant_norm in zip(uncategorized_indices, uncategorized_merchants):
            category = llm_results.get(merchant_norm, "Other")
            transactions[idx]["category"] = category
            cache_category(merchant_norm, category, source="llm")

    return transactions


def llm_categorize_batch(merchants: list[str]) -> dict[str, str]:
    """One LLM call for ALL unmatched merchants, strict-JSON response expected."""
    unique_merchants = list(dict.fromkeys(merchants))  # de-dupe, preserve order

    prompt = f"""You are a transaction categorizer. Categorize each merchant name below into EXACTLY ONE of these categories:
{", ".join(CATEGORIES)}

Merchants to categorize:
{json.dumps(unique_merchants)}

Respond with ONLY a JSON object mapping each merchant string exactly as given to one category from the list above.
No explanation, no markdown, no code fences. Just the raw JSON object."""

    try:
        response_text = call_ollama(prompt)
        cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE).strip()
        result = json.loads(cleaned)
        return {
            m: (result.get(m) if result.get(m) in CATEGORIES else "Other")
            for m in unique_merchants
        }
    except Exception as e:
        print(f"[categorizer] LLM batch categorization failed, defaulting to 'Other': {e}")
        return {m: "Other" for m in unique_merchants}
