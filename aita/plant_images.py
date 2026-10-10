"""A photo for every plant card, from Wikimedia Commons, with the credit its licence asks for.

Usage:
    python -m aita.plant_images        # fetches missing photos into web/images/plants/,
                                       # copies them to demo/photos/ (test images for scanning),
                                       # and writes the credits to data/plant_images.json

Only freely licensed Commons files are used; anything marked non-free is skipped.
Standard library only, so the static site build needs nothing installed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aita.knowledge import Plant, load_plants

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "web" / "images" / "plants"
DEMO_DIR = ROOT / "demo" / "photos"
CREDITS_PATH = ROOT / "data" / "plant_images.json"
USER_AGENT = "AitarKotha/0.1 (https://github.com/jemankalita/aitar-kotha)"
THUMB_WIDTH = 800
PAUSE_SECONDS = 0.5  # be gentle with Wikimedia's API

# Wikipedia article titles where the scientific name is not the best page (or has no good photo).
TITLE_OVERRIDES = {
    "kol": "Banana",
    "bah": "Bamboo",
    "xoriyoh": "Brassica juncea",
    "chah": "Camellia sinensis",
    "dhan": "Rice",
    "kaji_nemu": "Lemon",
    "bhut_jolokia": "Bhut jolokia",
}

_TAGS = re.compile(r"<[^>]+>")


def strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAGS.sub("", value))).strip()


def wiki_title(plant: Plant) -> str:
    return TITLE_OVERRIDES.get(plant.id, plant.scientific)


def _first_page(response: Mapping[str, Any]) -> Mapping[str, Any]:
    pages = response.get("query", {}).get("pages", {})
    return next(iter(pages.values()), {})


def parse_page_image(response: Mapping[str, Any]) -> tuple[str, str] | None:
    """(thumbnail URL, Commons file name) of an article's lead image, if it has one."""
    page = _first_page(response)
    thumb = page.get("thumbnail", {}).get("source")
    name = page.get("pageimage")
    return (thumb, name) if thumb and name else None


def parse_commons_credit(response: Mapping[str, Any]) -> dict[str, str] | None:
    """Author, licence and source page of a Commons file; None if missing or not freely licensed."""
    infos = _first_page(response).get("imageinfo") or []
    if not infos:
        return None
    meta = infos[0].get("extmetadata", {})
    value = lambda key: strip_html(str(meta.get(key, {}).get("value", "")))  # noqa: E731
    licence = value("LicenseShortName")
    if not licence or value("NonFree").lower() == "true":
        return None
    return {
        "author": value("Artist") or "Unknown",
        "license": licence,
        "license_url": value("LicenseUrl"),
        "source": infos[0].get("descriptionurl", ""),
    }


def load_image_credits(plants: Iterable[Plant], path: Path = CREDITS_PATH) -> dict[str, dict[str, str]]:
    if not Path(path).is_file():
        return {}
    known = {p.id for p in plants}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {plant_id: credit for plant_id, credit in raw.items() if plant_id in known}


def image_view(credit: Mapping[str, str] | None) -> dict[str, str] | None:
    """What the web app needs to show a photo and its credit (paths are relative for GitHub Pages)."""
    if not credit:
        return None
    return {**credit, "src": f"images/plants/{credit['file']}"}


# ---------- fetching (network) ----------


def _get_json(url: str, params: Mapping[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())


def fetch_one(plant: Plant) -> dict[str, str] | None:
    page = _get_json(
        "https://en.wikipedia.org/w/api.php",
        {"action": "query", "format": "json", "prop": "pageimages", "piprop": "thumbnail|name",
         "pithumbsize": str(THUMB_WIDTH), "redirects": "1", "titles": wiki_title(plant)},
    )
    found = parse_page_image(page)
    if not found:
        return None
    thumb_url, file_name = found
    commons = _get_json(
        "https://commons.wikimedia.org/w/api.php",
        {"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "extmetadata|url",
         "titles": f"File:{file_name}"},
    )
    credit = parse_commons_credit(commons)
    if not credit:
        return None
    suffix = Path(urllib.parse.urlparse(thumb_url).path).suffix.lower() or ".jpg"
    file = f"{plant.id}{suffix}"
    _download(thumb_url, IMAGES_DIR / file)
    return {"file": file, **credit}


def fetch_all(plants: Sequence[Plant]) -> dict[str, dict[str, str]]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    credits = load_image_credits(plants)
    for plant in plants:
        if plant.id in credits and (IMAGES_DIR / credits[plant.id]["file"]).is_file():
            continue
        try:
            credit = fetch_one(plant)
        except OSError as exc:
            print(f"  {plant.id}: could not fetch ({exc})")
            continue
        print(f"  {plant.id}: {credit['license'] + ' · ' + credit['author'][:40] if credit else 'no free photo found'}")
        if credit:
            credits[plant.id] = credit
        time.sleep(PAUSE_SECONDS)
    for credit in credits.values():
        shutil.copy2(IMAGES_DIR / credit["file"], DEMO_DIR / credit["file"])
    CREDITS_PATH.write_text(json.dumps(credits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return credits


def main(argv: Sequence[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    plants = load_plants()
    credits = fetch_all(plants)
    print(f"{len(credits)} of {len(plants)} plants have a photo")


if __name__ == "__main__":
    main()
