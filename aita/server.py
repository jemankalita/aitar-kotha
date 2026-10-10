"""The walking app's backend: ask Aita by name, by photo, or by season.

Run:
    uvicorn aita.server:build_default_app --factory --host 0.0.0.0 --port 8000

Every answer falls back to the family notes when a model is missing or down,
so the app still works on a trail with no signal.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from aita.knowledge import Plant, assamese_month, in_season, load_plants, load_translations, plant_by_id
from aita.llm import Endpoint, Identification, LLMError, chat, endpoint_from_env, identify_plant
from aita.persona import build_messages, compose_answer, season_answer, unknown_answer
from aita.retrieval import find_plants
from aita.voice import VoiceConfig, VoiceError, load_env_file, speech_text, synthesize, voice_from_env

logger = logging.getLogger("aita.server")

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DEFAULT_AUDIO_DIR = ROOT_DIR / "audio_cache"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_QUESTION_CHARS = 500
MIN_CONFIDENCE = 0.5
_SEASON_WORDS = re.compile(r"\b(month|season|this week|today|growing|flowering|in bloom|look for)\b", re.I)

ChatFn = Callable[..., str]
IdentifyFn = Callable[..., Identification]
SpeakFn = Callable[..., bytes]
PlantView = Callable[[Plant], dict[str, Any]]


class AskRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    month: int | None = Field(default=None, ge=1, le=12)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value.strip()


def _answer_about(
    question: str, context: Sequence[Plant], story: Endpoint | None, chat_fn: ChatFn
) -> tuple[str, str]:
    """(answer, source): the model's words when it is available, otherwise the family notes."""
    if story is not None:
        try:
            return chat_fn(story, build_messages(question, context)), "model"
        except LLMError as exc:
            logger.warning("story model unavailable, using family notes: %s", exc)
    if context:
        return compose_answer(context[0]), "notes"
    return unknown_answer(), "notes"


def _register_reads(
    app: FastAPI,
    plants: tuple[Plant, ...],
    view: PlantView,
    story: Endpoint | None,
    vision: Endpoint | None,
    voice: VoiceConfig | None,
) -> None:
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "story": story is not None,
            "vision": vision is not None,
            "voice": voice is not None,
            "plants": len(plants),
        }

    @app.get("/api/plants")
    def list_plants() -> list[dict[str, Any]]:
        return [view(p) for p in plants]

    @app.get("/api/season")
    def season(month: int = Query(ge=1, le=12)) -> dict[str, Any]:
        return {
            "month": month,
            "assamese_month": assamese_month(month),
            "plants": [view(p) for p in in_season(plants, month)],
        }


def _register_ask(
    app: FastAPI, plants: tuple[Plant, ...], view: PlantView, story: Endpoint | None, chat_fn: ChatFn
) -> None:
    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        found = find_plants(request.question, plants)
        if not found and _SEASON_WORDS.search(request.question):
            month = request.month or date.today().month
            return {"answer": season_answer(month, plants), "plant": None, "source": "notes", "kind": "season"}
        answer, source = _answer_about(request.question, found, story, chat_fn)
        plant = view(found[0]) if found else None
        return {"answer": answer, "plant": plant, "source": source, "kind": "plant" if found else "unknown"}


def _sniff_image_type(data: bytes) -> str | None:
    """The real image type from the file's first bytes; the client's Content-Type is not trusted."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def _read_image(image: UploadFile) -> tuple[bytes, str]:
    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="That photo is too big; try a smaller one.")
    mime_type = _sniff_image_type(data)
    if mime_type is None:
        raise HTTPException(status_code=415, detail="Please send a JPEG, PNG, WebP or GIF photo.")
    return data, mime_type


def _register_identify(
    app: FastAPI,
    plants: tuple[Plant, ...],
    view: PlantView,
    story: Endpoint | None,
    vision: Endpoint | None,
    chat_fn: ChatFn,
    identify_fn: IdentifyFn,
) -> None:
    # A plain `def` so FastAPI runs the slow model call in its thread pool, not on the event loop.
    @app.post("/api/identify")
    def identify(image: UploadFile = File(...)) -> dict[str, Any]:
        if vision is None:
            raise HTTPException(status_code=503, detail="No vision model is set up. Ask Aita by name instead.")
        data, mime_type = _read_image(image)
        try:
            verdict = identify_fn(vision, data, mime_type, plants)
        except LLMError as exc:
            logger.warning("vision model failed: %s", exc)
            raise HTTPException(status_code=502, detail="The vision model did not answer. Ask Aita by name.") from exc
        guess = plant_by_id(plants, verdict.plant_id) if verdict.plant_id else None
        result = {"confidence": verdict.confidence, "reason": verdict.reason}
        if guess is None or verdict.confidence < MIN_CONFIDENCE:
            unsure = "I'm not sure, child. Get closer to a leaf or a flower and show me again."
            if guess is not None:
                unsure = f"Maybe {guess.assamese} ({guess.roman})? I'm not sure. Look closer and show me again."
            return {**result, "plant": None, "guess": view(guess) if guess else None, "answer": unsure, "source": "notes"}
        answer, source = _answer_about(f"What is {guess.roman}?", (guess,), story, chat_fn)
        return {**result, "plant": view(guess), "guess": None, "answer": answer, "source": source}


def _register_speak(
    app: FastAPI,
    plants: tuple[Plant, ...],
    assamese: Mapping[str, dict[str, Any]],
    voice: VoiceConfig | None,
    speak_fn: SpeakFn,
    audio_dir: Path,
) -> None:
    # A plain `def`: the ElevenLabs call is slow and blocking, so FastAPI runs it in its thread pool.
    @app.get("/api/speak/{plant_id}")
    def speak(plant_id: str, lang: str = Query("as", pattern="^(as|en)$")) -> Response:
        if voice is None:
            raise HTTPException(status_code=503, detail="Aita's voice is not set up (no ElevenLabs key).")
        try:
            plant = plant_by_id(plants, plant_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Aita doesn't know that plant.") from None
        text = speech_text(plant, assamese.get(plant.id), lang)
        # Same words + same voice = same clip, so each one is paid for only once.
        digest = hashlib.sha256(f"{voice.model_id}|{voice.voice_id}|{lang}|{text}".encode()).hexdigest()
        cached = audio_dir / f"{digest}.mp3"
        if cached.is_file():
            return Response(cached.read_bytes(), media_type="audio/mpeg")
        try:
            audio = speak_fn(voice, text, lang)
        except VoiceError as exc:
            logger.warning("voice failed for %s: %s", plant_id, exc)
            raise HTTPException(status_code=502, detail="Aita's voice is not available right now.") from None
        audio_dir.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(audio)
        return Response(audio, media_type="audio/mpeg")


def create_app(
    plants: Sequence[Plant],
    story: Endpoint | None = None,
    vision: Endpoint | None = None,
    chat_fn: ChatFn = chat,
    identify_fn: IdentifyFn = identify_plant,
    translations: Mapping[str, dict[str, Any]] | None = None,
    voice: VoiceConfig | None = None,
    speak_fn: SpeakFn = synthesize,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
) -> FastAPI:
    plant_tuple = tuple(plants)
    assamese = dict(translations or {})

    def view(plant: Plant) -> dict[str, Any]:
        return {**plant.to_dict(), "as": assamese.get(plant.id)}

    app = FastAPI(title="Aitar Kotha", description="An Assamese plant field guide that talks like your grandmother.")
    _register_reads(app, plant_tuple, view, story, vision, voice)
    _register_speak(app, plant_tuple, assamese, voice, speak_fn, audio_dir)
    _register_ask(app, plant_tuple, view, story, chat_fn)
    _register_identify(app, plant_tuple, view, story, vision, chat_fn, identify_fn)
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def build_default_app() -> FastAPI:
    """Factory for uvicorn: reads the plant notes and model endpoints from the environment."""
    logging.basicConfig(level=logging.INFO)
    load_env_file(ROOT_DIR / ".env")
    story = endpoint_from_env("STORY")
    vision = endpoint_from_env("VISION")
    voice = voice_from_env()
    logger.info(
        "story model: %s | vision model: %s | voice: %s", story and story.model, vision and vision.model, voice
    )
    plants = load_plants()
    translations = load_translations(plants=plants)
    return create_app(plants, story=story, vision=vision, translations=translations, voice=voice)
