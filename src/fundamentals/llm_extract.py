from __future__ import annotations

import json
import os
from typing import Type, TypeVar

import requests
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _extract_with_openai(system_prompt: str, user_text: str, model_cls: Type[T]) -> T:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("no OpenAI key")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    truncated = user_text[:120000]
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": truncated},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)
    return model_cls.model_validate(data)


def _extract_with_ollama(system_prompt: str, user_text: str, model_cls: Type[T]) -> T:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    url = f"{base}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_text[:120000]},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2},
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    body = resp.json()
    raw = body.get("message", {}).get("content") or "{}"
    data = json.loads(raw) if isinstance(raw, str) else raw
    return model_cls.model_validate(data)


def extract_json_with_llm(system_prompt: str, user_text: str, model_cls: Type[T]) -> T:
    """
    Resolution order:
    1) OPENAI_API_KEY -> OpenAI JSON mode (cloud).
    2) USE_OLLAMA=1 -> local Ollama /api/chat with format=json (no API key).
    3) Empty defaults.

    Note: Cursor's in-IDE model is not callable from this Python process; use Ollama locally instead.
    """
    if not user_text.strip():
        return model_cls()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    use_ollama = os.getenv("USE_OLLAMA", "0").lower() in ("1", "true", "yes", "y")

    if api_key:
        try:
            return _extract_with_openai(system_prompt, user_text, model_cls)
        except Exception:
            pass

    if use_ollama:
        try:
            return _extract_with_ollama(system_prompt, user_text, model_cls)
        except Exception:
            pass

    return model_cls()
