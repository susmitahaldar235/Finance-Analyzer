"""
graph_state.py
Shared state passed between nodes in the LangGraph multi-agent workflow.

LangGraph passes one state object through the whole graph -- every node
reads what it needs from it and writes back updates. This is the "shared
memory" all four agents (Orchestrator, Categorization, Insight, Q&A) read
from and write to as the graph executes.

Two flows through the same graph:
  - "upload" flow:  transactions -> Categorization Agent -> Insight Agent -> END
  - "question" flow: question -> Q&A Agent -> END
`mode` is what the Orchestrator's routing function checks to decide which
path to take.
"""

from typing import TypedDict, Literal, Optional


class FinanceGraphState(TypedDict, total=False):
    # --- set by the caller before invoking the graph ---
    mode: Literal["upload", "question"]

    # --- upload flow fields ---
    raw_transactions: list[dict]           # parsed from CSV, not yet categorized
    categorized_transactions: list[dict]   # written by Categorization Agent
    uncertain_count: int                   # how many transactions the Categorization Agent flagged as low-confidence
    stats_snapshot: dict                   # computed stats (totals, MoM, recurring, anomalies) for the Insight Agent to reason over
    summary: str                           # written by Insight Agent -- final plain-English summary
    highlighted_insights: list[str]        # written by Insight Agent -- the 2-3 things it decided actually matter

    # --- question flow fields ---
    question: str                          # the user's free-text question
    answer: str                            # written by Q&A Agent
    tool_calls_made: list[dict]            # which tools the Q&A Agent chose to call, for transparency/demo purposes

    # --- routing metadata (for demo/debugging -- shows which agents ran) ---
    agent_trace: list[str]
