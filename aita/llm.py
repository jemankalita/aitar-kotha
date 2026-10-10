"""Talk to any OpenAI-compatible endpoint: local Ollama/llama.cpp, or Tinker's sampler.

Two roles, both configured with environment variables:
  AITA_STORY_*   the fine-tuned "Aita" model that answers in her voice
  AITA_VISION_*  an open vision model (e.g. Gemma 4 on Ollama) that picks the plant in a photo
Each role needs <ROLE>_BASE_URL and <ROLE>_MODEL; <ROLE>_API_KEY is optional.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from aita.knowledge import Plant

DEFAULT_TIMEOUT_SECONDS = 90.0
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable."""


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key: str | None = None


@dataclass(frozen=True)
class Identification:
    plant_id: str | None
    confidence: float
    reason: str


def endpoint_from_env(role: str) -> Endpoint | None:
    base_url = os.environ.get(f"AITA_{role}_BASE_URL", "").strip()
    model = os.environ.get(f"AITA_{role}_MODEL", "").strip()
    if not base_url or not model:
        return None
    api_key = os.environ.get(f"AITA_{role}_API_KEY", "").strip() or None
    return Endpoint(base_url=base_url.rstrip("/"), model=model, api_key=api_key)


def _post_chat(endpoint: Endpoint, payload: dict[str, Any], client: httpx.Client | None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    url = f"{endpoint.base_url}/chat/completions"
    owned = client is None
    http = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        response = http.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise LLMError(f"could not reach model at {endpoint.base_url}: {exc}") from exc
    finally:
        if owned:
            http.close()
    if response.status_code != 200:
        raise LLMError(f"model returned HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as exc:
        raise LLMError("model response was not JSON") from exc


def _message_text(body: dict[str, Any]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("unexpected response shape from model") from exc
    if not isinstance(content, str):
        raise LLMError("model response had no text content")
    return _THINK_BLOCK.sub("", content).strip()


def chat(
    endpoint: Endpoint,
    messages: Sequence[dict[str, Any]],
    *,
    max_tokens: int = 400,
    temperature: float = 0.4,
    client: httpx.Client | None = None,
) -> str:
    payload = {
        "model": endpoint.model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return _message_text(_post_chat(endpoint, payload, client))


def _candidate_list(plants: Sequence[Plant]) -> str:
    return "\n".join(f"- {p.id}: {p.english} ({p.scientific}). {p.look_for}" for p in plants)


def _identify_prompt(plants: Sequence[Plant]) -> str:
    return (
        "Which plant from this list is in the photo? Only choose from the list.\n"
        f"{_candidate_list(plants)}\n\n"
        'Reply with JSON only: {"id": "<id from the list or unknown>", "confidence": <0 to 1>, '
        '"why": "<the visible features you used>"}'
    )


def parse_identification(text: str, plants: Sequence[Plant]) -> Identification:
    """Pull the JSON verdict out of a vision model reply; anything off-list becomes unknown."""
    known = {p.id for p in plants}
    for match in _JSON_OBJECT.finditer(text):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        plant_id = data.get("id") if data.get("id") in known else None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return Identification(plant_id, min(max(confidence, 0.0), 1.0), str(data.get("why", "")))
    return Identification(None, 0.0, "")


def identify_plant(
    endpoint: Endpoint,
    image: bytes,
    mime_type: str,
    plants: Sequence[Plant],
    *,
    client: httpx.Client | None = None,
) -> Identification:
    data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _identify_prompt(plants)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    text = chat(endpoint, messages, max_tokens=200, temperature=0.0, client=client)
    return parse_identification(text, plants)
