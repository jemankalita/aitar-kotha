"""Turn the family plant notes into a chat fine-tuning set for Tinker.

Usage:
    python -m aita.dataset            # writes data/sft/train.jsonl and data/sft/eval.jsonl

Training questions and evaluation questions use different phrasings, so the eval
measures whether the model learned the plants, not whether it memorised the prompts.
"""

from __future__ import annotations

import argparse
import calendar
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from aita.knowledge import Plant, load_plants
from aita.persona import build_messages, compose_answer, season_answer, unknown_answer

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sft"
FIELD_NOTES_RATE = 0.5  # half the examples show the notes in context, half rely on memory

TRAIN_TEMPLATES = (
    "Aita, what is {roman}?",
    "Tell me about {roman}.",
    "What is {english} called in Assamese?",
    "What is the Assamese name for {scientific}?",
    "{assamese} কি?",
    "I found some {roman} on my walk. What can we do with it?",
    "Does {english} grow in Assam? What do people use it for?",
)
DESCRIBE_TEMPLATE = "Aita, I saw this on my walk: {look_for} What is it?"
SEASON_TEMPLATE = "What should I look for on my walk in {month}?"

EVAL_TEMPLATES = (
    "Koka told me to look for {roman} today. What is it?",
    "What do Assamese people call {english}?",
    "{assamese}ৰ বিষয়ে কোৱাচোন।",
    "What is {scientific} in Assamese?",
)

NOT_FROM_HERE = (
    "saguaro cactus",
    "edelweiss",
    "Douglas fir",
    "giant sequoia",
    "sugar maple",
    "baobab",
    "Japanese cherry blossom",
    "Venus flytrap",
    "Australian eucalyptus",
    "Dutch tulip",
    "Provence lavender",
    "Greek olive tree",
)
UNKNOWN_TEMPLATES = ("Aita, what is a {name}?", "Tell me about the {name}.")
SAFETY_QUESTIONS = (
    "Can I eat this wild mushroom I found by the road?",
    "These shiny black berries look tasty. Can I eat them?",
    "I found a plant I have never seen. Is it safe to chew the leaves?",
)


def _fill(template: str, plant: Plant) -> str:
    return template.format(
        roman=plant.roman,
        english=plant.english,
        scientific=plant.scientific,
        assamese=plant.assamese,
        look_for=plant.look_for,
    )


def _example(question: str, answer: str, context: Sequence[Plant]) -> dict[str, Any]:
    messages = build_messages(question, context)
    return {"messages": [*messages, {"role": "assistant", "content": answer}]}


def _plant_examples(plants: Sequence[Plant], rng: random.Random) -> list[dict[str, Any]]:
    examples = []
    for p_index, plant in enumerate(plants):
        templates = (*TRAIN_TEMPLATES, DESCRIBE_TEMPLATE)
        for t_index, template in enumerate(templates):
            context = (plant,) if rng.random() < FIELD_NOTES_RATE else ()
            answer = compose_answer(plant, variant=p_index + t_index)
            examples.append(_example(_fill(template, plant), answer, context))
    return examples


def _season_examples(plants: Sequence[Plant]) -> list[dict[str, Any]]:
    return [
        _example(SEASON_TEMPLATE.format(month=calendar.month_name[m]), season_answer(m, plants), ())
        for m in range(1, 13)
    ]


def _humility_examples() -> list[dict[str, Any]]:
    questions = [t.format(name=name) for name in NOT_FROM_HERE for t in UNKNOWN_TEMPLATES]
    return [_example(q, unknown_answer(), ()) for q in (*questions, *SAFETY_QUESTIONS)]


def build_training_set(plants: Sequence[Plant], seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    examples = [
        *_plant_examples(plants, rng),
        *_season_examples(plants),
        *_humility_examples(),
    ]
    rng.shuffle(examples)
    return examples


def build_eval_set(plants: Sequence[Plant]) -> list[dict[str, str]]:
    return [
        {"question": _fill(template, plant), "expected_id": plant.id, "kind": "plant"}
        for plant in plants
        for template in EVAL_TEMPLATES
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    plants = load_plants()
    n_train = write_jsonl(args.out_dir / "train.jsonl", build_training_set(plants, args.seed))
    n_eval = write_jsonl(args.out_dir / "eval.jsonl", build_eval_set(plants))
    print(f"{len(plants)} plants -> {n_train} training chats, {n_eval} held-out eval questions in {args.out_dir}")


if __name__ == "__main__":
    main()
