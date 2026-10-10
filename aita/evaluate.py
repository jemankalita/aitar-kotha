"""Compare the base model with fine-tuned Aita on questions she never saw in training.

Usage:
    python -m aita.evaluate                # uses runs/latest.json and data/sft/eval.jsonl
    python -m aita.evaluate --limit 40     # quicker, cheaper run

Writes reports/eval.md (a table for the DEV post) and reports/eval.json (every answer).
Neither model gets the field notes here: this measures what the weights learned.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from aita.dataset import DEFAULT_OUT_DIR, read_jsonl
from aita.knowledge import load_plants, plant_by_id
from aita.persona import build_messages
from aita.scoring import score_answer, summarize
from aita.train_tinker import DEFAULT_RUN_FILE, ROOT

DEFAULT_REPORT_DIR = ROOT / "reports"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_METRIC_LABELS = (
    ("assamese", "Assamese script name"),
    ("roman", "Roman Assamese name"),
    ("scientific", "Scientific name"),
)
EXAMPLES_IN_REPORT = 5


def message_text(message: dict[str, Any]) -> str:
    """Visible text of a parsed message, skipping any thinking parts."""
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return _THINK_BLOCK.sub("", str(content)).strip()


def comparison_table(base: dict[str, float], tuned: dict[str, float]) -> str:
    lines = ["| Metric | Base model | Aita (fine-tuned) |", "|---|---|---|"]
    lines += [f"| {label} | {base[key]:.0%} | {tuned[key]:.0%} |" for key, label in _METRIC_LABELS]
    lines.append(f"\nHeld-out questions: {tuned['n']}")
    return "\n".join(lines)


def _resolve(value: Any) -> Any:
    return value.result() if hasattr(value, "result") else value


def _tokenizer(sampling_client: Any, model: str) -> Any:
    getter = getattr(sampling_client, "get_tokenizer", None)
    if callable(getter):
        return getter()
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer(model)


def sample_answers(sampling_client: Any, renderer: Any, questions: Sequence[str], max_tokens: int) -> list[str]:
    import tinker

    params = tinker.types.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=renderer.get_stop_sequences())
    pending = [
        sampling_client.sample(
            prompt=renderer.build_generation_prompt(build_messages(q, ())), num_samples=1, sampling_params=params
        )
        for q in questions
    ]
    answers = []
    for future in pending:
        message, _ = renderer.parse_response(_resolve(future).sequences[0].tokens)
        answers.append(message_text(message))
    return answers


def _report(rows: list[dict[str, Any]], base: dict[str, float], tuned: dict[str, float]) -> str:
    parts = ["# Aita vs the base model\n", comparison_table(base, tuned), "\n## Examples\n"]
    for row in rows[:EXAMPLES_IN_REPORT]:
        parts += [f"**Q:** {row['question']}\n", f"*Base:*\n\n{row['base']}\n", f"*Aita:*\n\n{row['tuned']}\n", "---\n"]
    return "\n".join(parts)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import tinker
    from tinker_cookbook.renderers import get_renderer

    run = json.loads(args.run_file.read_text(encoding="utf-8"))
    plants = load_plants()
    eval_rows = read_jsonl(args.eval)[: args.limit] if args.limit else read_jsonl(args.eval)
    questions = [row["question"] for row in eval_rows]

    service = tinker.ServiceClient()
    base_client = service.create_sampling_client(base_model=run["base_model"])
    tuned_client = service.create_sampling_client(model_path=run["tinker_path"])
    renderer = get_renderer(run["renderer"], _tokenizer(base_client, run["base_model"]))

    print(f"Sampling {len(questions)} held-out questions from base and fine-tuned models...")
    base_answers = sample_answers(base_client, renderer, questions, args.max_tokens)
    tuned_answers = sample_answers(tuned_client, renderer, questions, args.max_tokens)

    rows = []
    for row, base_answer, tuned_answer in zip(eval_rows, base_answers, tuned_answers):
        plant = plant_by_id(plants, row["expected_id"])
        rows.append({
            **row,
            "base": base_answer,
            "tuned": tuned_answer,
            "base_score": score_answer(base_answer, plant),
            "tuned_score": score_answer(tuned_answer, plant),
        })
    base = summarize(r["base_score"] for r in rows)
    tuned = summarize(r["tuned_score"] for r in rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval.json").write_text(
        json.dumps({"base": base, "tuned": tuned, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "eval.md").write_text(_report(rows, base, tuned), encoding="utf-8")
    print(comparison_table(base, tuned))
    return {"base": base, "tuned": tuned}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-file", type=Path, default=DEFAULT_RUN_FILE)
    parser.add_argument("--eval", type=Path, default=DEFAULT_OUT_DIR / "eval.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N questions (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
