"""
main.py
FastAPI application tying everything together.

Routes:
    GET  /health                 -> check Ollama is reachable
    POST /upload                 -> upload+parse+convert currency+run multi-agent graph+store
    GET  /overview                -> full dashboard bundle (totals, MoM, recurring, anomalies)
    GET  /summary/{month}/stream  -> streamed plain-English summary for a month (lightweight, single-shot -- separate from the multi-agent graph)
    POST /ask                     -> free-text question answered via the multi-agent graph's Q&A Agent
    POST /reset                   -> clear all stored data (demo convenience)

Multi-agent architecture (see orchestrator_graph.py):
    /upload routes through: Categorization Agent -> Insight Agent
    /ask     routes through: Q&A Agent
Both flows share one LangGraph StateGraph, dispatched by mode.

Run with:
    uvicorn main:app --reload --port 7864
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from database import init_db, insert_transactions, clear_all_data
from csv_parser import parse_csv
from stats_engine import overview_summary, category_totals, month_over_month
from ollama_client import stream_ollama, check_ollama_available
from currency import init_currency_table, convert_amount, BASE_CURRENCY
from orchestrator_graph import run_upload_flow, run_question_flow

app = FastAPI(title="Personal Finance Analyzer")

# Locked-down CORS: only allow the local frontend origin, not "*".
# This matters because the backend listens on localhost and we don't want
# an arbitrary malicious webpage able to hit it via a drive-by script.
_allowed_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:7864",
    "http://127.0.0.1:7864",
    "null",]
# EXTRA_CORS_ORIGIN lets you add one more origin (e.g. a LAN IP or a hosted
# frontend URL) without editing this file -- set it as an env var if needed.
if os.environ.get("EXTRA_CORS_ORIGIN"):
    _allowed_origins.append(os.environ["EXTRA_CORS_ORIGIN"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    init_currency_table()


@app.get("/health")
def health():
    ollama_ok = check_ollama_available()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_reachable": ollama_ok,
        "message": None if ollama_ok else (
            "Ollama isn't reachable at 127.0.0.1:11434. "
            "Run `ollama serve` and make sure the model is pulled."
        ),
    }


@app.post("/upload")
def upload_statement(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported. See README for scope notes on PDF support.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        transactions, warnings = parse_csv(tmp_path, source_filename=file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not transactions:
        return {"inserted": 0, "warnings": warnings}

    # Currency conversion is deterministic pre-processing (not an agent's job
    # -- there's nothing to reason about, just apply the day's exchange rate)
    # so it happens before the transactions ever reach the multi-agent graph.
    foreign_currency_count = 0
    for txn in transactions:
        if txn["original_currency"] != BASE_CURRENCY:
            foreign_currency_count += 1
        txn["amount"] = convert_amount(txn["original_amount"], txn["date"], txn["original_currency"], BASE_CURRENCY)

    # Hand off to the multi-agent graph: Categorization Agent -> Insight Agent.
    # See orchestrator_graph.py for the routing, agents/ for each agent's logic.
    graph_result = run_upload_flow(transactions)
    categorized = graph_result["categorized_transactions"]

    insert_transactions(categorized)

    months_touched = sorted(set(t["month"] for t in categorized))
    if foreign_currency_count:
        warnings.append(
            f"Converted {foreign_currency_count} transaction(s) from a foreign currency to {BASE_CURRENCY}."
        )
    if graph_result.get("uncertain_count"):
        warnings.append(
            f"{graph_result['uncertain_count']} transaction(s) were categorized with low confidence -- "
            f"worth double-checking."
        )

    return {
        "inserted": len(categorized),
        "months_touched": months_touched,
        "warnings": warnings,
        "highlighted_insights": graph_result.get("highlighted_insights", []),
        "agent_summary": graph_result.get("summary", ""),
        "agent_trace": graph_result.get("agent_trace", []),
    }


@app.get("/overview")
def overview():
    return overview_summary()


@app.get("/summary/{month}/stream")
def stream_summary(month: str):
    """Streams a plain-English summary of a month's spending, built from
    already-computed stats (not raw transactions) -- keeps the LLM's job
    to 'explain the numbers', not 'discover the numbers'. This is a
    lightweight single-shot endpoint, separate from the Insight Agent's
    summary (which additionally decides *which* findings matter, not just
    describes all of them) -- kept here for a fast, always-available
    streaming demo of the dashboard on load."""
    totals = category_totals(month)
    mom = month_over_month(month)

    if not totals:
        raise HTTPException(404, f"No data for month {month}")

    prompt = f"""Here is a user's spending data for {month}:

Category totals: {totals}

Comparison to previous month: {mom}

Write a short, friendly, plain-English summary (3-5 sentences) of their spending this month.
Mention the biggest category, any notable increases or decreases vs last month, and one
practical observation. Do not use markdown formatting, just plain sentences."""

    def token_stream():
        for token in stream_ollama(prompt):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    # Hand off to the multi-agent graph's Q&A Agent (see agents/qna_agent.py).
    # Unlike the categorization/insight agents (which have their own internal
    # try/except fallbacks), the ReAct agent's underlying LLM call can raise
    # directly if Ollama is unreachable -- caught here so the user gets a
    # clean, actionable error instead of a raw stack trace.
    try:
        result = run_question_flow(payload.question)
    except Exception as e:
        raise HTTPException(
            503,
            f"Could not reach the local model to answer this question. "
            f"Check that Ollama is running (see /health). Details: {e}"
        )
    return {
        "answer": result.get("answer", ""),
        "tool_calls_made": result.get("tool_calls_made", []),
        "agent_trace": result.get("agent_trace", []),
    }


@app.post("/reset")
def reset():
    clear_all_data()
    return {"status": "cleared"}
