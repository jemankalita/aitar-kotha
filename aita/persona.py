"""Aita's voice: the system prompt, the field-notes block, and offline template answers.

The template answers double as fine-tuning targets, so the model learns both the facts
and the shape of an answer: Assamese name first, how to know it, how we use it, a story,
a caution, and then a push to put the phone away and go outside.
"""

from __future__ import annotations

import calendar
from typing import Iterable, Sequence

from aita.knowledge import Plant, assamese_month, in_season

SYSTEM_PROMPT = (
    "You are Aita, an Assamese grandmother who knows the plants of Assam. "
    "When asked about a plant, start with its Assamese name in Assamese script and in roman letters, "
    "then the English and scientific names. Say how to recognise it, how families use it, and one story "
    "or custom. Always mention any danger. Keep it short: the walk matters more than the screen. "
    "If you do not know a plant, say so plainly and send the person to ask an elder; never guess and "
    "never tell anyone to eat something they cannot name for sure. "
    "End by sending the person back outside to look, touch or smell something."
)

_LOOK_PREFIXES = ("How to know it:", "Look for this:", "You will know it like this:")
_USE_PREFIXES = ("We use it like this:", "In our house:", "What we do with it:")
_STORY_PREFIXES = ("", "Listen:", "I remember:")
_CLOSINGS = (
    "Now put the phone away and go find one.",
    "Go and look for it yourself, child; the phone can wait.",
    "Close the phone, go outside, and touch its leaves.",
)


def _header(plant: Plant) -> str:
    return f"{plant.assamese} — {plant.roman} ({plant.english}, {plant.scientific})."


def _join(prefix: str, text: str) -> str:
    return f"{prefix} {text}".strip()


def compose_answer(plant: Plant, variant: int = 0) -> str:
    """A short answer in Aita's voice, built only from the family notes."""
    v = variant % len(_CLOSINGS)
    lines = [
        _header(plant),
        _join(_LOOK_PREFIXES[v], plant.look_for),
        plant.season_note,
        _join(_USE_PREFIXES[v], " ".join(plant.uses)),
        _join(_STORY_PREFIXES[v], plant.story),
    ]
    if plant.caution:
        lines.append(f"Careful: {plant.caution}")
    lines.append(_CLOSINGS[v])
    return "\n".join(lines)


def facts_block(plant: Plant) -> str:
    """The family notes for one plant, formatted for the model's context window."""
    status = "checked with elders" if plant.verified else "family notes, not yet checked with elders"
    lines = [
        f"Field notes ({status}):",
        f"- Assamese: {plant.assamese} ({plant.roman})",
        f"- English: {plant.english}",
        f"- Scientific: {plant.scientific}",
        f"- Season: {plant.season_note}",
        f"- Look for: {plant.look_for}",
        f"- Uses: {' '.join(plant.uses)}",
        f"- Story: {plant.story}",
    ]
    if plant.caution:
        lines.append(f"- Caution: {plant.caution}")
    return "\n".join(lines)


def build_messages(question: str, context: Sequence[Plant]) -> list[dict[str, str]]:
    """Chat messages for the model, with field notes appended when retrieval found plants."""
    system = SYSTEM_PROMPT
    if context:
        system = "\n\n".join([SYSTEM_PROMPT, *(facts_block(p) for p in context)])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def unknown_answer() -> str:
    return (
        "I don't know that one, child. It is not from our garden or our hills, and I won't guess. "
        "Ask an elder who knows it, or look in a good field guide, and never eat something you cannot "
        "name for sure. Then go and look at it again, and come and tell me what you learned."
    )


def season_answer(month: int, plants: Iterable[Plant]) -> str:
    """What to look for on a walk this month."""
    local = assamese_month(month)
    english = calendar.month_name[month]
    found = in_season(plants, month)
    if not found:
        return f"It is {local} ({english}). I have no special plants written down for this month. Go and see what is out there."
    lines = [f"It is {local} ({english}) now. On your walk, look for:"]
    lines.extend(f"- {p.assamese} ({p.roman}): {p.english}" for p in found)
    lines.append("Go out and see how many you can find before dark.")
    return "\n".join(lines)
