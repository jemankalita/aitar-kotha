"""Load and validate the family plant notes in data/plants.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "plants.json"
DEFAULT_TRANSLATIONS_PATH = DEFAULT_PATH.with_name("plants_as.json")
_TRANSLATED_TEXT = ("season_note", "look_for", "story")

# The Assamese month that begins (around the middle of) each Gregorian month.
ASSAMESE_MONTHS = {
    1: "Magh",
    2: "Phagun",
    3: "Sot",
    4: "Bohag",
    5: "Jeth",
    6: "Ahar",
    7: "Xaon",
    8: "Bhado",
    9: "Ahin",
    10: "Kati",
    11: "Aghon",
    12: "Puh",
}

_REQUIRED_TEXT = (
    "id",
    "assamese",
    "roman",
    "english",
    "scientific",
    "kind",
    "season_note",
    "look_for",
    "story",
)


class KnowledgeError(ValueError):
    """Raised when the plant notes are malformed."""


@dataclass(frozen=True)
class Plant:
    id: str
    assamese: str
    roman: str
    aliases: tuple[str, ...]
    english: str
    scientific: str
    kind: str
    months: tuple[int, ...]
    season_note: str
    look_for: str
    uses: tuple[str, ...]
    story: str
    caution: str | None
    verified: bool

    def names(self) -> tuple[str, ...]:
        """Every way someone might name this plant, lowercased (Assamese script is unchanged)."""
        return (
            self.roman.lower(),
            self.assamese,
            self.english.lower(),
            self.scientific.lower(),
            *self.aliases,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "assamese": self.assamese,
            "roman": self.roman,
            "aliases": list(self.aliases),
            "english": self.english,
            "scientific": self.scientific,
            "kind": self.kind,
            "months": list(self.months),
            "season_note": self.season_note,
            "look_for": self.look_for,
            "uses": list(self.uses),
            "story": self.story,
            "caution": self.caution,
            "verified": self.verified,
        }


def _text(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeError(f"plant {raw.get('id')!r}: field {field!r} must be non-empty text")
    return value.strip()


def _text_list(raw: dict[str, Any], field: str) -> tuple[str, ...]:
    value = raw.get(field, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise KnowledgeError(f"plant {raw.get('id')!r}: field {field!r} must be a list of text")
    return tuple(v.strip() for v in value)


def _months(raw: dict[str, Any]) -> tuple[int, ...]:
    value = raw.get("months", [])
    if not isinstance(value, list) or not all(isinstance(m, int) and 1 <= m <= 12 for m in value):
        raise KnowledgeError(f"plant {raw.get('id')!r}: field 'months' must be numbers 1-12")
    return tuple(value)


def parse_plant(raw: dict[str, Any]) -> Plant:
    """Validate one raw record and return an immutable Plant."""
    if not isinstance(raw, dict):
        raise KnowledgeError("each plant must be a JSON object")
    text = {field: _text(raw, field) for field in _REQUIRED_TEXT}
    caution = raw.get("caution")
    if caution is not None and (not isinstance(caution, str) or not caution.strip()):
        raise KnowledgeError(f"plant {text['id']!r}: 'caution' must be text or null")
    return Plant(
        **text,
        aliases=tuple(a.lower() for a in _text_list(raw, "aliases")),
        months=_months(raw),
        uses=_text_list(raw, "uses"),
        caution=caution.strip() if caution else None,
        verified=bool(raw.get("verified", False)),
    )


def load_plants(path: Path | str = DEFAULT_PATH) -> tuple[Plant, ...]:
    """Load every plant from a JSON file, rejecting malformed or duplicate entries."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(f"could not read plant notes from {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise KnowledgeError("plant notes must be a JSON array")
    plants = tuple(parse_plant(item) for item in raw)
    seen: set[str] = set()
    for plant in plants:
        if plant.id in seen:
            raise KnowledgeError(f"duplicate plant id {plant.id!r}")
        seen.add(plant.id)
    return plants


def _parse_translation(plant_id: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise KnowledgeError(f"translation for {plant_id!r} must be a JSON object")
    record = {"id": plant_id, **raw}
    caution = raw.get("caution")
    if caution is not None and (not isinstance(caution, str) or not caution.strip()):
        raise KnowledgeError(f"translation for {plant_id!r}: 'caution' must be text or null")
    return {
        **{field: _text(record, field) for field in _TRANSLATED_TEXT},
        "uses": list(_text_list(record, "uses")),
        "caution": caution.strip() if caution else None,
    }


def load_translations(
    path: Path | str = DEFAULT_TRANSLATIONS_PATH, *, plants: Iterable[Plant]
) -> dict[str, dict[str, Any]]:
    """Assamese versions of the notes, keyed by plant id. A missing file means none yet."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(f"could not read translations from {path}: {exc}") from exc
    entries = raw.get("plants") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        raise KnowledgeError("translations file must have a 'plants' object")
    known = {p.id for p in plants}
    unknown = sorted(set(entries) - known)
    if unknown:
        raise KnowledgeError(f"translations for unknown plants: {', '.join(unknown)}")
    return {plant_id: _parse_translation(plant_id, entry) for plant_id, entry in entries.items()}


def plant_by_id(plants: Iterable[Plant], plant_id: str) -> Plant:
    for plant in plants:
        if plant.id == plant_id:
            return plant
    raise KeyError(plant_id)


def _check_month(month: int) -> None:
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")


def in_season(plants: Iterable[Plant], month: int) -> tuple[Plant, ...]:
    """Plants worth looking for in a given Gregorian month."""
    _check_month(month)
    return tuple(p for p in plants if month in p.months)


def assamese_month(month: int) -> str:
    _check_month(month)
    return ASSAMESE_MONTHS[month]
