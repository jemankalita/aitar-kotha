import json

import httpx
import pytest

from aita.knowledge import load_plants
from aita.llm import Endpoint, LLMError, chat, endpoint_from_env, identify_plant, parse_identification

PLANTS = load_plants()
ENDPOINT = Endpoint(base_url="http://local.test/v1", model="aita", api_key="secret")


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_chat_posts_openai_style_request_and_returns_text():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _reply("বগৰী — bogori")

    text = chat(ENDPOINT, [{"role": "user", "content": "hi"}], client=_client(handler))

    assert text == "বগৰী — bogori"
    assert seen["url"] == "http://local.test/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "aita"


def test_chat_strips_thinking_blocks():
    text = chat(ENDPOINT, [], client=_client(lambda r: _reply("<think>hmm</think>\nবগৰী")))

    assert text == "বগৰী"


def test_chat_raises_llm_error_on_http_failure():
    with pytest.raises(LLMError, match="500"):
        chat(ENDPOINT, [], client=_client(lambda r: httpx.Response(500, text="boom")))


def test_chat_raises_llm_error_on_connection_failure():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="reach"):
        chat(ENDPOINT, [], client=_client(handler))


def test_chat_raises_llm_error_on_malformed_body():
    with pytest.raises(LLMError, match="response"):
        chat(ENDPOINT, [], client=_client(lambda r: httpx.Response(200, json={"oops": 1})))


def test_parse_identification_accepts_known_id():
    result = parse_identification('Sure! {"id": "bogori", "confidence": 0.8, "why": "thorny"}', PLANTS)

    assert result.plant_id == "bogori"
    assert result.confidence == 0.8


def test_parse_identification_rejects_unknown_ids_and_garbage():
    assert parse_identification('{"id": "maple", "confidence": 0.9}', PLANTS).plant_id is None
    assert parse_identification("no json here", PLANTS).plant_id is None


def test_parse_identification_clamps_confidence():
    assert parse_identification('{"id": "bogori", "confidence": 7}', PLANTS).confidence == 1.0


def test_identify_plant_sends_image_and_candidates():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _reply('{"id": "sewali", "confidence": 0.7, "why": "orange tubes"}')

    result = identify_plant(ENDPOINT, b"\x89PNG", "image/png", PLANTS, client=_client(handler))

    assert result.plant_id == "sewali"
    content = seen["body"]["messages"][-1]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    text_part = next(part for part in content if part["type"] == "text")
    assert "bogori" in text_part["text"]


def test_endpoint_from_env_requires_url_and_model(monkeypatch):
    monkeypatch.delenv("AITA_STORY_BASE_URL", raising=False)
    monkeypatch.delenv("AITA_STORY_MODEL", raising=False)
    assert endpoint_from_env("STORY") is None

    monkeypatch.setenv("AITA_STORY_BASE_URL", "http://x/v1/")
    monkeypatch.setenv("AITA_STORY_MODEL", "m")
    monkeypatch.setenv("AITA_STORY_API_KEY", "k")
    assert endpoint_from_env("STORY") == Endpoint(base_url="http://x/v1", model="m", api_key="k")
