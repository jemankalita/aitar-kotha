"""Find which plants a question is about, by name, alias, script or near-miss spelling."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

from aita.knowledge import Plant

# Keep Latin letters, digits and the Bengali-Assamese Unicode block; everything else is a space.
_NON_WORD = re.compile(r"[^a-z0-9ঀ-৿]+")
FUZZY_THRESHOLD = 0.84
MIN_FUZZY_LENGTH = 4


def normalize(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def _exact_score(padded_question: str, plant: Plant) -> int:
    """Length of the longest name that appears as whole words in the question (0 if none)."""
    best = 0
    for name in plant.names():
        key = normalize(name)
        if key and f" {key} " in padded_question:
            best = max(best, len(key))
    return best


def _fuzzy_score(tokens: list[str], plant: Plant) -> float:
    """Best similarity between a single question word and a single-word name of the plant."""
    best = 0.0
    single_word_names = [n for n in map(normalize, plant.names()) if " " not in n and len(n) >= MIN_FUZZY_LENGTH]
    for token in tokens:
        if len(token) < MIN_FUZZY_LENGTH:
            continue
        for name in single_word_names:
            best = max(best, SequenceMatcher(None, token, name).ratio())
    return best


def find_plants(question: str, plants: Iterable[Plant], limit: int = 3) -> tuple[Plant, ...]:
    """Plants named in the question, best match first. Exact names beat fuzzy spellings."""
    normalized = normalize(question)
    padded = f" {normalized} "
    tokens = normalized.split()
    scored: list[tuple[float, Plant]] = []
    for plant in plants:
        exact = _exact_score(padded, plant)
        if exact:
            scored.append((100.0 + exact, plant))
            continue
        fuzzy = _fuzzy_score(tokens, plant)
        if fuzzy >= FUZZY_THRESHOLD:
            scored.append((fuzzy, plant))
    scored.sort(key=lambda item: item[0], reverse=True)
    return tuple(plant for _, plant in scored[:limit])
