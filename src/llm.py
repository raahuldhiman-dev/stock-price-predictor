"""LLM client with provider fallback: Groq first, then Gemini.

Keys come from environment variables (or a local .env file), never from code:
  GROQ_API_KEY, GEMINI_API_KEY
Optional overrides: GROQ_MODEL, GEMINI_MODEL
"""
import os

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TIMEOUT = 30


class LLMError(Exception):
    pass


def _groq(prompt, system):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        raise LLMError(f"Groq HTTP {r.status_code}")
    return r.json()["choices"][0]["message"]["content"].strip()


def _gemini(prompt, system):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3},
        },
        timeout=TIMEOUT,
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
