"""Build the static demo site (for GitHub Pages): the web app plus the family notes, no AI at runtime.

Usage:
    python -m aita.build_static                 # writes dist/

The demo answers from data/plants.json in the browser. The open-model parts (fine-tuned Aita,
photo scanning) run in the local version; the demo says so and links to the repo.
Voice clips made with `python -m aita.voice_clips` are shipped if voice_clips/ has any.
Standard library only, so the deploy job needs nothing installed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

from aita.knowledge import load_plants, load_translations
from aita.plant_images import image_view, load_image_credits

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DEFAULT_OUT = ROOT / "dist"
DEFAULT_CLIPS = ROOT / "voice_clips"
REPO_URL = "https://github.com/jemankalita/aitar-kotha"


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def build_site(out_dir: Path, *, clips_dir: Path = DEFAULT_CLIPS, repo_url: str = REPO_URL) -> Path:
    out_dir = Path(out_dir).resolve()
    if out_dir in (WEB_DIR.resolve(), ROOT.resolve()) or WEB_DIR.resolve().is_relative_to(out_dir):
        raise ValueError(f"refusing to build into {out_dir}: it would delete the source")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(WEB_DIR, out_dir)

    plants = load_plants()
    translations = load_translations(plants=plants)
    photos = load_image_credits(plants)
    _write_json(
        out_dir / "plants.json",
        [{**p.to_dict(), "as": translations.get(p.id), "image": image_view(photos.get(p.id))} for p in plants],
    )

    clips = sorted(Path(clips_dir).glob("*.mp3")) if Path(clips_dir).is_dir() else []
    if clips:
        (out_dir / "audio").mkdir()
        for clip in clips:
            shutil.copy2(clip, out_dir / "audio" / clip.name)
    _write_json(out_dir / "config.json", {"mode": "demo", "repo": repo_url, "voice": bool(clips)})
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")  # serve files as-is on GitHub Pages
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    args = parser.parse_args(argv)
    out = build_site(args.out, clips_dir=args.clips)
    print(f"Static demo site written to {out}")


if __name__ == "__main__":
    main()
