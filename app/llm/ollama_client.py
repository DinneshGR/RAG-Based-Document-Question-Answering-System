"""
Ollama client module.

Talks to a locally running Ollama server via its REST API. No API key
is required. The base URL and model are fully configurable through
environment variables (OLLAMA_BASE_URL, OLLAMA_MODEL) so the LLM can be
swapped without touching any other code.
"""
from __future__ import annotations

import logging

import requests

from app.config import settings
from app.exceptions import ModelUnavailableError, OllamaUnavailableError

logger = logging.getLogger(__name__)


def check_ollama_health() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def is_model_available(model_name: str | None = None) -> bool:
    """Check whether the configured model has been pulled into Ollama."""
    model_name = model_name or settings.OLLAMA_MODEL
    try:
        resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        available_models = [m["name"] for m in resp.json().get("models", [])]
        # Ollama tags can include ":latest" implicitly; compare loosely.
        return any(
            model_name == m or model_name == m.split(":")[0] or m.startswith(model_name)
            for m in available_models
        )
    except requests.exceptions.RequestException:
        return False


def generate_answer(system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
    """Call Ollama's /api/chat endpoint and return the assistant's text reply."""
    if not check_ollama_health():
        raise OllamaUnavailableError(
            "Could not connect to Ollama. Make sure it is installed and running "
            f"at {settings.OLLAMA_BASE_URL} (start it with `ollama serve`)."
        )

    if not is_model_available():
        raise ModelUnavailableError(
            f"Model '{settings.OLLAMA_MODEL}' is not available in Ollama. "
            f"Pull it first with: ollama pull {settings.OLLAMA_MODEL}"
        )

    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE},
    }

    try:
        resp = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=settings.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise OllamaUnavailableError(f"Failed to generate answer from Ollama: {exc}") from exc

    data = resp.json()
    message = data.get("message", {})
    content = message.get("content", "").strip()

    if not content:
        raise ModelUnavailableError("Ollama returned an empty response.")

    logger.info("Generated answer using model=%s", settings.OLLAMA_MODEL)
    return content
