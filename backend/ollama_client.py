"""
ollama_client.py
Thin wrapper around Ollama's local HTTP API (http://127.0.0.1:11434).

Prereqs before this works:
    ollama pull qwen2.5:7b
    ollama serve      (usually auto-starts after install, but run explicitly if unsure)

Why Qwen2.5-instruct: reliable structured/JSON output and solid function-calling
support in Ollama at a size (7B, or 1.5B for low-spec machines) that runs on
CPU without a GPU. See project notes for full reasoning -- swap MODEL_NAME
below if you prefer a different local model; everything else stays the same
as long as it supports Ollama's /api/chat format.
"""

import os
import requests
import json

# Locally this defaults to Ollama running on the host machine. Inside Docker,
# docker-compose sets OLLAMA_HOST=http://ollama:11434 so the backend container
# reaches the ollama container by its service name over the Docker network.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:7b")   


def call_ollama(prompt: str, system: str = None) -> str:
    """Single non-streaming call. Used where we need one clean text/JSON blob back
    (e.g. batch categorization)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1}   # low temp -> consistent categorization, not creative writing
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def stream_ollama(prompt: str, system: str = None):
    """Generator yielding tokens as they arrive. Used for the natural-language
    summary endpoint so the frontend can render text as it streams in, instead
    of waiting for the full response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    with requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.4}
    }, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break


def check_ollama_available() -> bool:
    """Quick health check used by /health endpoint so the UI can show a clear
    'backend model not running' message instead of a silent failure."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False
