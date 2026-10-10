"""Aita's voice: read a plant card aloud with ElevenLabs, in Assamese or English.

This is the one closed piece of the project and it is optional: with no ELEVENLABS_API_KEY
the app hides the Listen button and everything else keeps working.

Environment (or a local .env file, never committed):
    ELEVENLABS_API_KEY   required to enable the voice
    AITA_VOICE_ID        which ElevenLabs voice speaks as Aita (default: Sarah, soft female)
    AITA_VOICE_MODEL     default eleven_v4 (90+ languages, including Assamese)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

from aita.knowledge import Plant

API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah: soft female; change it to the voice that sounds most like Aita
DEFAULT_MODEL = "eleven_v4"
TIMEOUT_SECONDS = 60.0
LANGUAGES = ("as", "en")


class VoiceError(RuntimeError):
    """ElevenLabs could not be reached or refused the request. Never contains the API key."""


@dataclass(frozen=True)
class VoiceConfig:
    api_key: str
    voice_id: str = DEFAULT_VOICE_ID
    model_id: str = DEFAULT_MODEL

    def __repr__(self) -> str:  # keep the key out of logs and tracebacks
        return f"VoiceConfig(voice_id={self.voice_id!r}, model_id={self.model_id!r})"


def load_env_file(path: Path | str) -> None:
    """Read KEY=value lines from a .env file into os.environ without overriding what is already set."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def voice_from_env() -> VoiceConfig | None:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        return None
    return VoiceConfig(
        api_key=api_key,
        voice_id=os.environ.get("AITA_VOICE_ID", "").strip() or DEFAULT_VOICE_ID,
        model_id=os.environ.get("AITA_VOICE_MODEL", "").strip() or DEFAULT_MODEL,
    )


def speech_text(plant: Plant, assamese_notes: Mapping[str, Any] | None, lang: str) -> str:
    """What Aita says aloud: the name, her story, and any warning. Short enough to hear on a walk."""
    if lang == "as" and assamese_notes:
        parts = [f"{plant.assamese}।", assamese_notes["story"]]
        if assamese_notes.get("caution"):
            parts.append(f"সাৱধান: {assamese_notes['caution']}")
        return " ".join(parts)
    parts = [f"{plant.roman}, {plant.english}.", plant.story]
    if plant.caution:
        parts.append(f"Careful: {plant.caution}")
    return " ".join(parts)


def synthesize(config: VoiceConfig, text: str, lang: str, *, client: httpx.Client | None = None) -> bytes:
    """Turn text into MP3 bytes with ElevenLabs text-to-speech."""
    owned = client is None
    http = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        response = http.post(
            f"{API_URL}/{config.voice_id}",
            params={"output_format": OUTPUT_FORMAT},
            headers={"xi-api-key": config.api_key, "accept": "audio/mpeg"},
            json={"text": text, "model_id": config.model_id, "language_code": lang},
        )
    except httpx.HTTPError as exc:
        raise VoiceError(f"could not reach ElevenLabs: {type(exc).__name__}") from None
    finally:
        if owned:
            http.close()
    if response.status_code != 200:
        raise VoiceError(f"ElevenLabs returned HTTP {response.status_code}: {response.text[:200]}")
    return response.content
