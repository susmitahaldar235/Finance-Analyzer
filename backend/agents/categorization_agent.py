"""
agents/categorization_agent.py
Specialist agent responsible for categorizing transactions.

Reuses the existing two-tier categorizer (cache -> rules -> batched LLM
fallback, see categorizer.py) as its core logic, and adds one genuinely
agentic step on top: for transactions the rules couldn't match and the LLM
had to guess on, it asks the LLM a second time to self-assess confidence,
and flags low-confidence ones instead of silently accepting a guess. That
self-check is the agent's own decision, not a fixed rule.
"""

import json
import re
from langchain_ollama import ChatOllama
from categorizer import categorize_transactions, CATEGORIES, RULES
from graph_state import FinanceGraphState

MODEL_NAME = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _confidence_check(merchant: str, assigned_category: str, llm: ChatOllama) -> bool:
    """Ask the LLM to self-assess whether its own categorization is confident.
    Returns True if confident, False if it should be flagged for the user."""
    prompt = f"""A transaction from merchant "{merchant}" was categorized as "{assigned_category}".
Is this a confident, sensible categorization, or a guess that could easily be wrong?
Respond with ONLY one word: CONFIDENT or UNSURE."""
    try:
        response = llm.invoke(prompt)
        return "CONFIDENT" in response.content.upper()
    except Exception as e:
        print(f"[categorization_agent] confidence check failed, defaulting to confident: {e}")
        return True


def categorization_agent_node(state: FinanceGraphState) -> dict:
    """
    LangGraph node. Reads state['raw_transactions'], writes
    state['categorized_transactions'] and state['uncertain_count'].
    """
    transactions = state["raw_transactions"]

    # Core categorization -- unchanged two-tier logic (cache -> rules -> batched LLM fallback)
    categorized = categorize_transactions(transactions)

    # Agentic self-check step: only re-examine transactions that came from the
    # LLM fallback (rule-matched ones are already high-confidence by design).
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.1)
    uncertain_count = 0
    for txn in categorized:
        # We only spend the extra confidence-check call on transactions that
        # weren't matched by a keyword rule -- keeps this fast for the common case.
        was_rule_matched = any(
            kw in txn["merchant_normalized"]
            for keywords in RULES.values()
            for kw in keywords
        )
        if was_rule_matched:
            txn["confidence"] = "high"
            continue

        confident = _confidence_check(txn["merchant"], txn["category"], llm)
        txn["confidence"] = "high" if confident else "low"
        if not confident:
            uncertain_count += 1

    trace = state.get("agent_trace", [])
    trace.append("categorization_agent")

    return {
        "categorized_transactions": categorized,
        "uncertain_count": uncertain_count,
        "agent_trace": trace,
    }
