"""
agents/qna_agent.py
Specialist agent that answers the user's free-text questions about their
spending, deciding for itself which read-only data tools to call.

This is the same responsibility as the original hand-rolled agent.py, now
reimplemented using LangGraph's prebuilt create_react_agent -- the standard
library pattern for "LLM + tools, looping until it has a final answer."
Functionally equivalent to a hand-rolled loop, but framework-managed:
LangGraph handles the message bookkeeping, tool-call routing, and looping
that agent.py did manually.
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from stats_engine import category_totals, month_over_month, detect_recurring_charges, detect_anomalies
from database import get_distinct_months
from graph_state import FinanceGraphState

MODEL_NAME = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


# ---- Tools, defined with @tool so LangGraph can bind them directly ----
# Same 5 capabilities as the original agent.py -- just declared in the
# LangChain tool format instead of raw JSON schemas.

@tool
def get_category_totals_tool(month: str) -> dict:
    """Get total spend per category for a specific month (format YYYY-MM). Use this to answer questions about where money went in a given month."""
    return category_totals(month)


@tool
def get_month_over_month_tool(month: str) -> dict:
    """Compare a month's category spend against the previous month, with percentage changes. Use this for 'why did I spend more/less' type questions. Month format YYYY-MM."""
    return month_over_month(month)


@tool
def get_recurring_charges_tool() -> list:
    """Get all detected recurring charges / subscriptions across all uploaded months. Use this for questions about subscriptions or repeated charges."""
    return detect_recurring_charges()


@tool
def get_anomalies_tool(month: str) -> list:
    """Get spending categories that are statistically unusual (much higher/lower than historical average) for a given month, format YYYY-MM."""
    return detect_anomalies(month)


@tool
def get_available_months_tool() -> list:
    """List which months have uploaded data available. Use this first if you're unsure which months exist."""
    return get_distinct_months()


TOOLS = [
    get_category_totals_tool,
    get_month_over_month_tool,
    get_recurring_charges_tool,
    get_anomalies_tool,
    get_available_months_tool,
]


def _build_agent():
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.2)
    months = get_distinct_months()
    latest_month = months[-1] if months else None
    system_prompt = f"""You are a helpful personal finance assistant. You have access to tools that
query the user's actual transaction data -- always use them rather than guessing.
The latest month with data available is {latest_month or 'none yet'}.
When a user's question doesn't specify a month, assume they mean the latest available month.
After gathering the information you need via tools, give a clear, concise, plain-English answer,
referencing specific numbers and category names from the tool results."""
    return create_react_agent(llm, TOOLS, prompt=system_prompt)


def qna_agent_node(state: FinanceGraphState) -> dict:
    """
    LangGraph node. Reads state['question'], runs the prebuilt ReAct agent
    (which internally loops: LLM -> tool calls -> LLM -> ... -> final answer),
    and writes state['answer'] + state['tool_calls_made'] for transparency.
    """
    agent = _build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": state["question"]}]})

    messages = result["messages"]
    final_answer = messages[-1].content

    tool_calls_made = []
    for msg in messages:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            for call in calls:
                tool_calls_made.append({"tool": call["name"], "arguments": call["args"]})

    trace = state.get("agent_trace", [])
    trace.append("qna_agent")

    return {
        "answer": final_answer,
        "tool_calls_made": tool_calls_made,
        "agent_trace": trace,
    }
