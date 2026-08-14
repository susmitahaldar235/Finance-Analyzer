"""
agents/insight_agent.py
Specialist agent responsible for deciding what's actually worth telling the
user, and writing the plain-English summary.

Important boundary (say this in interviews): this agent does NOT compute any
numbers itself. All totals, percentage changes, recurring-charge detection,
and anomaly z-scores come from stats_engine.py, which is pure deterministic
Python. This agent's only job is to reason over already-computed stats and
decide which 2-3 things are significant enough to highlight, then explain
them in plain English. Keeping computation and reasoning separate means the
numbers are never hallucinated -- only the prioritization and phrasing are
left to the model.
"""

import json
import re
from langchain_ollama import ChatOllama
from stats_engine import category_totals, month_over_month, detect_recurring_charges, detect_anomalies
from graph_state import FinanceGraphState

MODEL_NAME = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def insight_agent_node(state: FinanceGraphState) -> dict:
    """
    LangGraph node. Reads the latest month from categorized_transactions,
    computes the full stats snapshot, and asks the LLM to decide what
    matters and summarize it. Writes state['stats_snapshot'],
    state['highlighted_insights'], and state['summary'].
    """
    categorized = state["categorized_transactions"]
    if not categorized:
        return {"summary": "No transactions to analyze.", "highlighted_insights": [], "stats_snapshot": {}}

    latest_month = max(t["month"] for t in categorized)

    stats_snapshot = {
        "month": latest_month,
        "category_totals": category_totals(latest_month),
        "month_over_month": month_over_month(latest_month),
        "recurring_charges": detect_recurring_charges(),
        "anomalies": detect_anomalies(latest_month),
    }

    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.3)

    prompt = f"""You are a financial insight assistant. Here is a user's computed spending data for {latest_month}:

{json.dumps(stats_snapshot, default=str, indent=2)}

Your job has two parts:
1. Decide which 2-3 findings are actually significant enough for the user to care about
   (large changes, unusual anomalies, notable recurring charges) -- not every number, just what matters.
2. Write a short, friendly, plain-English summary (3-5 sentences) covering only those significant findings.

Respond with ONLY a JSON object in this exact shape, no markdown, no code fences:
{{"highlighted_insights": ["short phrase 1", "short phrase 2"], "summary": "the plain-English paragraph"}}"""

    try:
        response = llm.invoke(prompt)
        cleaned = re.sub(r"^```json\s*|\s*```$", "", response.content.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        highlighted = parsed.get("highlighted_insights", [])
        summary = parsed.get("summary", "")
    except Exception as e:
        print(f"[insight_agent] LLM reasoning failed, falling back to raw stats: {e}")
        highlighted = []
        summary = f"Spending summary for {latest_month} is available, but the narrative summary could not be generated."

    trace = state.get("agent_trace", [])
    trace.append("insight_agent")

    return {
        "stats_snapshot": stats_snapshot,
        "highlighted_insights": highlighted,
        "summary": summary,
        "agent_trace": trace,
    }
