"""Pre-generate Aita's voice clips once, so the public demo plays them with no ElevenLabs key.

Usage (needs ELEVENLABS_API_KEY in .env):
    python -m aita.voice_clips          # writes voice_clips/<plant>-<as|en>.mp3, skips ones that exist

43 plants x 2 languages = 86 short clips. Commit voice_clips/ and the Pages build ships them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aita.knowledge import Plant, load_plants, load_translations
from aita.voice import LANGUAGES, VoiceConfig, load_env_file, speech_text, synthesize, voice_from_env

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "voice_clips"


def clip_name(plant_id: str, lang: str) -> str:
    return f"{plant_id}-{lang}.mp3"


def generate_clips(
    plants: Sequence[Plant],
    translations: Mapping[str, dict[str, Any]],
    config: VoiceConfig,
    out_dir: Path,
    *,
    speak_fn: Callable[..., bytes] = synthesize,
) -> int:
    """Make every missing clip; returns how many were made. Stops at the first ElevenLabs error."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for plant in plants:
        for lang in LANGUAGES:
            path = out_dir / clip_name(plant.id, lang)
            if path.exists():
                continue
            path.write_bytes(speak_fn(config, speech_text(plant, translations.get(plant.id), lang), lang))
            made += 1
    return made


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    load_env_file(ROOT / ".env")
    config = voice_from_env()
    if config is None:
        raise SystemExit("Set ELEVENLABS_API_KEY in .env first (see .env.example).")
    plants = load_plants()
    made = generate_clips(plants, load_translations(plants=plants), config, args.out)
    print(f"Made {made} new clips in {args.out}")


if __name__ == "__main__":
    main()
