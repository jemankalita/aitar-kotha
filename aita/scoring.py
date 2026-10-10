"""Score model answers: did it name the plant the way an Assamese family would?"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable

from aita.knowledge import Plant
from aita.retrieval import normalize

# Aliases this close to the roman name are spelling variants (xewali/sewali), not translations.
SPELLING_VARIANT_SIMILARITY = 0.7
FIELDS = ("assamese", "roman", "scientific")


def _roman_spellings(plant: Plant) -> tuple[str, ...]:
    roman = normalize(plant.roman)
    variants = (
        normalize(alias)
        for alias in plant.aliases
        if SequenceMatcher(None, normalize(alias), roman).ratio() >= SPELLING_VARIANT_SIMILARITY
    )
    return (roman, *variants)


def score_answer(answer: str, plant: Plant) -> dict[str, bool]:
    padded = f" {normalize(answer)} "
    return {
        "assamese": plant.assamese in answer,
        "roman": any(f" {spelling} " in padded for spelling in _roman_spellings(plant)),
        "scientific": plant.scientific.lower() in answer.lower(),
    }


def summarize(scores: Iterable[dict[str, bool]]) -> dict[str, float]:
    rows = list(scores)
    if not rows:
        return {**{field: 0.0 for field in FIELDS}, "n": 0}
    rates = {field: sum(row[field] for row in rows) / len(rows) for field in FIELDS}
    return {**rates, "n": len(rows)}
