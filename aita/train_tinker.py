"""Fine-tune an open-weight Qwen model into Aita with a LoRA adapter on Tinker.

Usage:
    export TINKER_API_KEY=...            # from the Tinker console (or `tinker auth login`)
    python -m aita.dataset               # build data/sft/train.jsonl first
    python -m aita.train_tinker          # trains, then writes runs/latest.json

Only the last assistant message of each chat is trained on, so the model learns Aita's
answers, not the questions. The run record points at the sampler weights, which the app
can use straight away through Tinker's OpenAI-compatible endpoint, or export to run locally.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from aita.dataset import DEFAULT_OUT_DIR, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_FILE = ROOT / "runs" / "latest.json"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_RENDERER = "qwen3_5"
LR_FLOOR_FRACTION = 0.1  # decay to 10% of the starting learning rate


def epoch_batches(n_items: int, batch_size: int, epochs: int, seed: int) -> list[list[int]]:
    """Index batches for every epoch; each epoch is a fresh deterministic shuffle."""
    if n_items <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("n_items, batch_size and epochs must all be positive")
    rng = random.Random(seed)
    batches: list[list[int]] = []
    for _ in range(epochs):
        order = list(range(n_items))
        rng.shuffle(order)
        batches.extend(order[i : i + batch_size] for i in range(0, n_items, batch_size))
    return batches


def learning_rate(step: int, total_steps: int, base_lr: float) -> float:
    """Linear decay from base_lr to LR_FLOOR_FRACTION * base_lr."""
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return base_lr * (1.0 - (1.0 - LR_FLOOR_FRACTION) * progress)


def run_record(tinker_path: str, base_model: str, renderer: str, steps: int, now: datetime) -> dict[str, Any]:
    return {
        "tinker_path": tinker_path,
        "base_model": base_model,
        "renderer": renderer,
        "steps": steps,
        "created_at": now.isoformat(),
    }


def _metrics_text(metrics: Any) -> str:
    if not isinstance(metrics, dict):
        return ""
    return " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()) if isinstance(v, (int, float)))


def _build_data(conversations: Sequence[list[dict[str, Any]]], renderer: Any, max_length: int) -> list[Any]:
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    return [
        conversation_to_datum(conv, renderer, max_length=max_length, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        for conv in conversations
    ]


def train(args: argparse.Namespace) -> dict[str, Any]:
    import tinker
    from tinker_cookbook.renderers import get_renderer

    conversations = [row["messages"] for row in read_jsonl(args.train)]
    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(base_model=args.model, rank=args.rank)
    renderer = get_renderer(args.renderer, training_client.get_tokenizer())
    data = _build_data(conversations, renderer, args.max_length)
    batches = epoch_batches(len(data), args.batch_size, args.epochs, args.seed)
    print(f"Training {args.model} (LoRA rank {args.rank}) on {len(data)} chats for {len(batches)} steps")

    for step, batch in enumerate(batches):
        lr = learning_rate(step, len(batches), args.lr)
        fwd_bwd = training_client.forward_backward([data[i] for i in batch], "cross_entropy")
        optim = training_client.optim_step(tinker.types.AdamParams(learning_rate=lr))
        result = fwd_bwd.result()
        optim.result()
        print(f"step {step + 1:>3}/{len(batches)}  lr={lr:.2e}  {_metrics_text(getattr(result, 'metrics', None))}")

    tinker_path = training_client.save_weights_for_sampler(args.name).result().path
    record = run_record(tinker_path, args.model, args.renderer, len(batches), datetime.now(timezone.utc))
    args.run_file.parent.mkdir(parents=True, exist_ok=True)
    args.run_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved sampler weights: {tinker_path}\nRun record: {args.run_file}")
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_OUT_DIR / "train.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--renderer", default=DEFAULT_RENDERER)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", default="aitar-kotha")
    parser.add_argument("--run-file", type=Path, default=DEFAULT_RUN_FILE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
