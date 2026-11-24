"""LLM client with provider fallback: Groq first, then Gemini.

Keys come from environment variables (or a local .env file), never from code:
  GROQ_API_KEY, GEMINI_API_KEY
Optional overrides: GROQ_MODEL, GEMINI_MODEL
"""
import os
import time

import requests

from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed: only real environment variables are used

TIMEOUT = 30


class LLMError(Exception):
    pass


def _post(url, retries=2, **kw):
    """POST with a short retry on rate limit / overload (429, 5xx)."""
    for attempt in range(retries + 1):
        r = requests.post(url, timeout=TIMEOUT, **kw)
        if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
            return r
        time.sleep(1.5 * (attempt + 1))


def _groq(prompt, system):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    r = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
    )
    if r.status_code != 200:
        raise LLMError(f"Groq HTTP {r.status_code}")
    return r.json()["choices"][0]["message"]["content"].strip()


def _gemini(prompt, system):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    r = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3},
        },
    )
    if r.status_code != 200:
        raise LLMError(f"Gemini HTTP {r.status_code}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


PROVIDERS = [("Groq", _groq), ("Gemini", _gemini)]
SYSTEM = (
    "You are a cautious financial-data analyst. Explain model outputs in plain language, "
    "note their limits, and never give investment advice or guarantees."
)


def ask(prompt, system=SYSTEM):
    """Return (text, provider_name). Tries each provider in order."""
    errors = []
    for name, fn in PROVIDERS:
        try:
            return fn(prompt, system), name
        except (LLMError, requests.RequestException, KeyError, IndexError, ValueError) as e:
            errors.append(f"{name}: {e}")
    raise LLMError("; ".join(errors))


def insight(prompt):
    """Streamlit-friendly wrapper: returns (text, provider) or (None, error)."""
    try:
        return ask(prompt)
    except LLMError as e:
        return None, str(e)
