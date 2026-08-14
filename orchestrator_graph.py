"""
orchestrator_graph.py
The Orchestrator: builds the LangGraph StateGraph that connects all four
agents and routes between them.

Two flows share this one graph, split by state['mode']:

    upload flow:    START -> route -> categorization_agent -> insight_agent -> END
    question flow:  START -> route -> qna_agent -> END

The routing function is plain Python (not an LLM call) -- it's a dispatcher,
not a decision-maker. The actual agentic reasoning happens inside each
specialist node (categorization_agent, insight_agent, qna_agent), where an
LLM decides what to do with the data it's given. This mirrors a common
LangGraph pattern: cheap deterministic routing at the graph level, LLM
reasoning inside the nodes that need it.
"""

from langgraph.graph import StateGraph, START, END

from graph_state import FinanceGraphState
from agents.categorization_agent import categorization_agent_node
from agents.insight_agent import insight_agent_node
from agents.qna_agent import qna_agent_node


def _route(state: FinanceGraphState) -> str:
    """Decides which path through the graph to take, based on state['mode'].
    This is deterministic dispatch, not agent reasoning -- kept intentionally
    simple and cheap since there's nothing to reason about here."""
    return "upload_path" if state["mode"] == "upload" else "question_path"


def build_graph():
    graph = StateGraph(FinanceGraphState)

    graph.add_node("categorization_agent", categorization_agent_node)
    graph.add_node("insight_agent", insight_agent_node)
    graph.add_node("qna_agent", qna_agent_node)

    # Conditional entry point: same graph, two different paths depending on mode.
    graph.add_conditional_edges(
        START,
        _route,
        {
            "upload_path": "categorization_agent",
            "question_path": "qna_agent",
        },
    )

    # Upload flow: categorization hands off to insight, then finishes.
    graph.add_edge("categorization_agent", "insight_agent")
    graph.add_edge("insight_agent", END)

    # Question flow: qna agent answers directly, then finishes.
    graph.add_edge("qna_agent", END)

    return graph.compile()


# Compiled once at import time and reused across requests -- compiling a
# LangGraph graph is a bit of setup work, no reason to redo it per-request.
finance_graph = build_graph()


def run_upload_flow(raw_transactions: list[dict]) -> dict:
    """Entry point used by main.py's /upload route."""
    result = finance_graph.invoke({
        "mode": "upload",
        "raw_transactions": raw_transactions,
        "agent_trace": [],
    })
    return result


def run_question_flow(question: str) -> dict:
    """Entry point used by main.py's /ask route."""
    result = finance_graph.invoke({
        "mode": "question",
        "question": question,
        "agent_trace": [],
    })
    return result
