import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aita.knowledge import load_plants, load_translations, plant_by_id
from aita.server import create_app
from aita.voice import (
    VoiceConfig,
    VoiceError,
    load_env_file,
    speech_text,
    synthesize,
    voice_from_env,
)

PLANTS = load_plants()
TRANSLATIONS = load_translations(plants=PLANTS)
BOGORI = plant_by_id(PLANTS, "bogori")
CONFIG = VoiceConfig(api_key="secret", voice_id="voice123", model_id="eleven_v4")
MP3 = b"ID3fake-mp3"


# ---------- what Aita says ----------


def test_assamese_speech_uses_assamese_notes():
    text = speech_text(BOGORI, TRANSLATIONS["bogori"], "as")

    assert text.startswith("বগৰী")
    assert "সৰস্বতী" in text  # the Saraswati Puja story, in Assamese
    assert "কাঁইট" in text  # the thorn warning


def test_english_speech_uses_english_notes():
    text = speech_text(BOGORI, None, "en")

    assert text.startswith("bogori")
    assert "Saraswati Puja" in text
    assert "Careful" in text


def test_assamese_without_translation_falls_back_to_english():
    assert "Saraswati Puja" in speech_text(BOGORI, None, "as")


# ---------- calling ElevenLabs ----------


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_synthesize_posts_to_elevenlabs_and_returns_audio():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("xi-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=MP3, headers={"content-type": "audio/mpeg"})

    audio = synthesize(CONFIG, "বগৰী", "as", client=_client(handler))

    assert audio == MP3
    assert seen["url"].startswith("https://api.elevenlabs.io/v1/text-to-speech/voice123")
    assert "output_format=mp3_44100_128" in seen["url"]
    assert seen["key"] == "secret"
    assert seen["body"] == {"text": "বগৰী", "model_id": "eleven_v4", "language_code": "as"}


def test_synthesize_raises_voice_error_on_http_failure():
    with pytest.raises(VoiceError, match="401"):
        synthesize(CONFIG, "x", "en", client=_client(lambda r: httpx.Response(401, text="bad key")))


def test_synthesize_raises_voice_error_when_unreachable():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(VoiceError, match="reach"):
        synthesize(CONFIG, "x", "en", client=_client(handler))


def test_synthesize_never_leaks_the_api_key_in_errors():
    with pytest.raises(VoiceError) as info:
        synthesize(CONFIG, "x", "en", client=_client(lambda r: httpx.Response(500, text="oops")))

    assert "secret" not in str(info.value)


# ---------- configuration ----------


def test_voice_from_env_needs_a_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert voice_from_env() is None

    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("AITA_VOICE_ID", "v")
    assert voice_from_env() == VoiceConfig(api_key="k", voice_id="v", model_id="eleven_v4")


def test_load_env_file_sets_missing_values_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nAITA_TEST_A=from-file\nAITA_TEST_B="quoted"\n\nnot a pair\n', encoding="utf-8")
    monkeypatch.setenv("AITA_TEST_A", "already-set")
    monkeypatch.delenv("AITA_TEST_B", raising=False)

    load_env_file(env)

    import os

    assert os.environ["AITA_TEST_A"] == "already-set"
    assert os.environ["AITA_TEST_B"] == "quoted"


def test_load_env_file_missing_is_fine(tmp_path):
    load_env_file(tmp_path / "nope.env")


# ---------- the /api/speak endpoint ----------


def _app(tmp_path, voice=CONFIG, speak_fn=None):
    calls = []

    def fake_speak(config, text, lang, **kwargs):
        calls.append((text, lang))
        return MP3

    app = create_app(
        PLANTS,
        translations=TRANSLATIONS,
        voice=voice,
        speak_fn=speak_fn or fake_speak,
        audio_dir=tmp_path / "audio",
    )
    return TestClient(app), calls


def test_speak_returns_mp3_in_assamese(tmp_path):
    client, calls = _app(tmp_path)

    response = client.get("/api/speak/bogori", params={"lang": "as"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == MP3
    assert calls[0][1] == "as"
    assert calls[0][0].startswith("বগৰী")


def test_speak_caches_audio_so_replays_cost_nothing(tmp_path):
    client, calls = _app(tmp_path)

    client.get("/api/speak/bogori", params={"lang": "as"})
    second = client.get("/api/speak/bogori", params={"lang": "as"})

    assert second.content == MP3
    assert len(calls) == 1


def test_speak_without_voice_config_is_503(tmp_path):
    client, _ = _app(tmp_path, voice=None)

    assert client.get("/api/speak/bogori").status_code == 503


def test_speak_unknown_plant_is_404(tmp_path):
    client, _ = _app(tmp_path)

    assert client.get("/api/speak/maple").status_code == 404


def test_speak_rejects_unknown_language(tmp_path):
    client, _ = _app(tmp_path)

    assert client.get("/api/speak/bogori", params={"lang": "fr"}).status_code == 422


def test_speak_upstream_failure_is_502_without_details(tmp_path):
    def broken(*args, **kwargs):
        raise VoiceError("ElevenLabs returned HTTP 401")

    client, _ = _app(tmp_path, speak_fn=broken)
    response = client.get("/api/speak/bogori")

    assert response.status_code == 502
    assert "401" not in response.text


def test_health_reports_voice(tmp_path):
    client, _ = _app(tmp_path)

    assert client.get("/api/health").json()["voice"] is True
