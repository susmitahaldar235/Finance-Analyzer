# Ledger — Local-First Personal Finance Analyzer

A privacy-first personal finance tool that parses bank/card CSV statements,
categorizes transactions, detects recurring charges and spending anomalies,
and answers free-text questions — powered by a **multi-agent LangGraph
workflow** running entirely on local LLMs via Ollama.

**Everything runs on your machine.** No bank-account linking (no Plaid), no
cloud LLM calls, no data leaving your computer — the only network calls are
to Ollama running on `127.0.0.1`.

---

## What it actually does

1. **Upload a CSV statement** → a **Categorization Agent** classifies each
   transaction (rule-based first, batched LLM fallback for unrecognized
   merchants, plus a self-confidence check that flags uncertain guesses),
   then hands off to an **Insight Agent** that decides which findings are
   actually worth surfacing and writes a plain-English summary.
2. **Dashboard** → category breakdown chart, month-over-month comparison,
   recurring-charge / subscription detection, and statistical anomaly flags —
   all computed with plain Python/stats, no LLM involved.
3. **Streamed plain-English summary** of the month's spending (a lightweight,
   always-available version separate from the Insight Agent's prioritized one).
4. **Ask a question** ("why did I spend more this month?", "what subscriptions
   am I paying for?") → a **Q&A Agent** decides which of 5 data-query tools to
   call, gathers results, and answers in plain English.
5. **Multi-currency support** → if your statement has a `currency` column
   (e.g. foreign travel spending), each transaction is automatically
   converted to your base currency (INR by default) using live historical
   exchange rates from the free Frankfurter API — the only external network
   call this project makes, and it only ever sends a currency pair + date,
   never any transaction amount or merchant name. The original amount and
   currency are kept alongside the converted value.

All four of the above (Categorization, Insight, Q&A, and the Orchestrator
that routes between them) are built as a **multi-agent LangGraph workflow**
— see the full architecture breakdown further down.

---

## Tech stack

| Layer               | Choice                          | Why |
|---------------------|----------------------------------|-----|
| Backend API         | FastAPI                          | Async, fast to build, good docs, native Pydantic validation |
| Multi-agent orchestration | LangGraph + LangChain-Ollama | State machine for routing between 4 agents; prebuilt `create_react_agent` for the Q&A agent's tool-calling loop |
| Local LLM runtime   | Ollama                           | Simplest way to run a local model with an HTTP API |
| Model               | Qwen2.5:7b (instruct)            | Reliable structured JSON output + function-calling support, runs on CPU |
| Data processing     | Pandas                           | CSV parsing, transaction handling |
| Storage             | SQLite (file-based)              | Zero setup, reinforces the "local-first" story, good enough at this scale |
| Currency conversion | Frankfurter API (free, no key)   | Live historical FX rates; only sends a currency pair + date, never transaction data |
| Frontend            | Single HTML file + vanilla JS    | No build step, fast to iterate; Chart.js for the category chart |

**Storage note:** this project needs two kinds of "storage" and both are covered:
- **Model weights** (Qwen2.5:7b via Ollama) — lives in Ollama's local model
  cache, ~4-5 GB on disk, no setup needed beyond `ollama pull`.
- **App data** (your transactions) — a single `finance.db` SQLite file created
  automatically in `backend/` on first run. Delete it any time to fully reset.

---

## Scope decisions (intentional, not oversights)

- **CSV only, not PDF.** Every bank lets you export CSV, and PDF layout
  parsing across different banks is the single biggest time-sink in a project
  like this for very little payoff. Noted here explicitly as a documented
  boundary, not a gap you forgot about.
- **Anomaly detection uses a simple z-score-style threshold**, not a full
  statistical model — appropriate for a personal tool, not a banking-grade system.
- **The LLM never invents numbers.** All totals, comparisons, and detections
  are computed in plain Python (`stats_engine.py`). The LLM's only jobs are:
  (a) categorize unmatched merchants, (b) explain already-computed numbers in
  plain English, (c) decide which read-only tool to call for a free-text question.

---

## Project structure

```
finance-analyzer/
├── backend/
│   ├── main.py                  FastAPI app + all routes
│   ├── orchestrator_graph.py     LangGraph StateGraph: routes between agents
│   ├── graph_state.py            shared state passed between agent nodes
│   ├── agents/
│   │   ├── categorization_agent.py   categorizes transactions + confidence self-check
│   │   ├── insight_agent.py          decides what's significant, writes summary
│   │   └── qna_agent.py              tool-calling agent for free-text Q&A (LangGraph create_react_agent)
│   ├── database.py               SQLite setup, queries
│   ├── csv_parser.py              CSV statement parsing
│   ├── categorizer.py             rule-based + LLM-fallback categorization (used by Categorization Agent)
│   ├── stats_engine.py            totals, month-over-month, recurring charges, anomalies (used by Insight + Q&A Agents)
│   ├── currency.py                multi-currency conversion via Frankfurter API
│   ├── ollama_client.py           thin wrapper around Ollama's HTTP API (used by categorizer.py + streaming summary)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── index.html                 upload UI + dashboard + chat, single file, no build step
├── sample_data/
│   ├── sample_statement.csv        3 months of realistic sample transactions
│   └── sample_multicurrency.csv    sample with USD/EUR transactions, for testing currency conversion
└── README.md                     (this file)
```

---

## Prerequisites

- **Python 3.10+**
- **Ollama** installed — [ollama.com/download](https://ollama.com/download)
- ~5 GB free disk space for the model
- 8 GB+ RAM recommended (see the low-spec note below if you have less)

---

## Setup — step by step

### 1. Install Ollama and pull the model

```bash
# Install Ollama (macOS/Linux) — or download the installer from ollama.com for Windows
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model (this downloads ~4-5 GB, only needed once)
ollama pull qwen2.5:7b
```

**Low-spec machine (< 8 GB RAM)?** Use the smaller model instead:
```bash
ollama pull qwen2.5:1.5b
```
Then open `backend/ollama_client.py` and change:
```python
MODEL_NAME = "qwen2.5:7b"
```
to:
```python
MODEL_NAME = "qwen2.5:1.5b"
```

### 2. Start Ollama (if it isn't already running)

```bash
ollama serve
```
On most installs Ollama auto-starts as a background service after install —
run this manually only if `ollama pull` or the app's health check says it
can't reach Ollama.

Leave this running in its own terminal tab.

### 3. Set up the Python backend

```bash
cd finance-analyzer/backend

# (recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```bash
uvicorn main:app --reload --port 7864
```

You should see something like:
```
INFO:     Uvicorn running on http://127.0.0.1:7864
INFO:     Application startup complete.
```

Leave this running in its own terminal tab too. Visit
`http://127.0.0.1:7864/health` in a browser — you should see
`"ollama_reachable": true` if steps 1-2 worked.

### 5. Open the frontend

Just open the file directly in your browser — no server needed for the frontend itself:

```bash
# macOS
open ../frontend/index.html

# Linux
xdg-open ../frontend/index.html

# Windows
start ../frontend/index.html
```

Or simply double-click `frontend/index.html` in your file explorer.

You should see the **Ledger** dashboard with a status pill in the top-right
showing whether the model backend is reachable.

### 6. Try it out

- Click **Upload**, select `sample_data/sample_statement.csv` (included, no
  need to find your own statement to try it out first)
- Watch the dashboard populate: category chart, month-over-month ledger,
  recurring charges, anomalies, and a streamed plain-English summary
- Try the chat box at the bottom: *"why did I spend more this month?"* or
  *"what subscriptions am I paying for?"* — watch which tools the agent
  chooses to call (shown under its answer)
- Use your own real CSV export any time — just make sure it has date,
  description, and amount columns (common header name variants are handled
  automatically, see `csv_parser.py`)

---

## Running order, every time

You need **three things running**, in this order:

1. `ollama serve` (if not already auto-started)
2. `uvicorn main:app --reload --port 7864` (from `backend/`)
3. Open `frontend/index.html` in your browser

---

## API reference (for testing with curl/Postman)

| Method | Route                        | Purpose |
|--------|-------------------------------|---------|
| GET    | `/health`                     | Check Ollama connectivity |
| POST   | `/upload`                     | Upload a CSV statement (multipart file) |
| GET    | `/overview`                   | Full dashboard data bundle |
| GET    | `/summary/{month}/stream`     | Streamed plain-English summary (e.g. `/summary/2026-06/stream`) |
| POST   | `/ask`                        | `{"question": "..."}` → agent answer + tools used |
| POST   | `/reset`                      | Clear all stored data |

Example:
```bash
curl -X POST http://127.0.0.1:7864/upload \
  -F "file=@../sample_data/sample_statement.csv"

curl http://127.0.0.1:7864/overview

curl -X POST http://127.0.0.1:7864/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"what subscriptions am I paying for?"}'
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Status pill shows "model offline" | Run `ollama serve`, confirm with `curl http://127.0.0.1:11434/api/tags` |
| Upload says "Could not reach backend" | Confirm uvicorn is running on port 7864 |
| Categorization always says "Other" | Ollama isn't reachable, or the model hasn't been pulled — check `/health` |
| CSV upload fails with "missing columns" | Your export needs a date, description/merchant, and amount column — rename headers or check `csv_parser.py`'s `COLUMN_ALIASES` |
| Agent gives generic answers, doesn't seem to use tools | Some smaller/older models are weaker at function-calling — try `qwen2.5:7b` instead of a smaller variant if you downgraded |
| Everything is slow | Expected on CPU-only inference for 7B models — try `qwen2.5:1.5b` for faster (lower quality) responses |
| Foreign-currency amounts look wrong / unconverted | The Frankfurter API may be briefly unreachable — the app falls back to a 1:1 rate rather than failing the upload, and flags this in the upload response warnings. Re-upload later, or check your internet connection |

---

## Multi-agent architecture (LangGraph)

This project uses **four agents**, orchestrated as a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph`. One shared graph handles two different flows, split by a `mode` field in the graph's state (see `graph_state.py`):

```
                         ┌─────────────────┐
                         │   Orchestrator   │   (plain routing function,
                         │  (route by mode) │    not an LLM call --
                         └────────┬─────────┘    see _route() in
                    ┌─────────────┴─────────────┐  orchestrator_graph.py)
                    │                             │
              mode="upload"                mode="question"
                    │                             │
                    ▼                             ▼
          ┌──────────────────┐           ┌──────────────────┐
          │ Categorization    │           │   Q&A Agent       │
          │ Agent             │           │ (LangGraph        │
          │ (categorizer.py + │           │  create_react_    │
          │  confidence check)│           │  agent, 5 tools)  │
          └────────┬──────────┘           └────────┬─────────┘
                    │                               │
                    ▼                               ▼
          ┌──────────────────┐                    END
          │  Insight Agent    │
          │ (decides what's   │
          │  worth surfacing, │
          │  writes summary)  │
          └────────┬──────────┘
                    │
                    ▼
                   END
```

**1. Orchestrator** (`orchestrator_graph.py`) — the entry point and router. Deliberately *not* an LLM call: given the graph is only ever in "upload" or "question" mode, there's nothing to reason about, so routing is a plain Python function. This mirrors a common LangGraph pattern — cheap deterministic dispatch at the graph level, LLM reasoning reserved for the nodes that actually need it.

**2. Categorization Agent** (`agents/categorization_agent.py`) — owns classifying transactions. Wraps the existing two-tier categorizer (cache → keyword rules → batched LLM fallback) and adds one genuinely agentic step: for any transaction the LLM had to guess on, it asks the model to self-assess its own confidence and flags low-confidence guesses instead of silently accepting them.

**3. Insight Agent** (`agents/insight_agent.py`) — owns deciding what's worth telling the user. Reads the computed stats snapshot (category totals, month-over-month, recurring charges, anomalies — all from `stats_engine.py`, pure deterministic Python) and decides which 2-3 findings are actually significant, then writes the plain-English summary. **Important boundary**: this agent never computes numbers itself, only reasons over and prioritizes numbers that were already computed elsewhere — so the model can misjudge *importance*, but it can never hallucinate a total or a percentage.

**4. Q&A Agent** (`agents/qna_agent.py`) — answers free-text questions. Built with LangGraph's prebuilt `create_react_agent`, the standard library pattern for "LLM + tools, loop until final answer." Given 5 read-only data tools, it decides which to call and in what order based on the question, exactly the same responsibility as before, just now framework-managed instead of a hand-rolled loop.

**Handoff, concretely:**
```
Upload flow:    CSV parsed → Categorization Agent categorizes + flags uncertain
                → hands categorized transactions to Insight Agent
                → Insight Agent computes significance + writes summary
                → response includes categorized data + highlighted insights + agent_trace

Question flow:  question → Q&A Agent picks tools, loops, answers
                → response includes the answer + which tools were called
```

Every `/upload` and `/ask` response includes an `agent_trace` field showing exactly which agents ran, in order — useful for demos and for answering "walk me through what happens when I upload a file" in an interview.

### Why LangGraph instead of hand-rolling the loop

An earlier version of this project (Q&A only) hand-rolled the tool-calling loop directly against Ollama's API. That's a legitimate, simpler approach for a single agent with a handful of tools — no framework overhead, full visibility into every message. LangGraph starts earning its keep once there are **multiple agents that need to hand off state to each other** (Categorization → Insight, in this version) — at that point, a graph-based state machine with defined nodes and edges is a clearer mental model than manually threading data through nested function calls, and it comes with built-in patterns (conditional routing, prebuilt agent constructors, retries) that would otherwise have to be hand-built.

---

## Possible next steps (explicitly out of scope for this build)

- PDF statement parsing (would need per-bank layout handling)
- Multi-bank CSV format auto-detection beyond the current alias system
- Proper anomaly detection (seasonal decomposition, not just z-scores)
- Multi-user support / auth (currently single-user, single local DB)
- Packaging as a desktop app (Electron/Tauri) so non-technical users don't
  need to run three separate processes manually
