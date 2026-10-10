"""Exercise train/evaluate/export end to end against fake tinker modules (no network, no API key)."""

import json
import sys
import types
from pathlib import Path

import pytest

from aita import evaluate, export_local, train_tinker
from aita.dataset import build_eval_set, build_training_set, write_jsonl
from aita.knowledge import load_plants

PLANTS = load_plants()


class _Done:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class FakeRenderer:
    def build_generation_prompt(self, messages):
        return messages[-1]["content"]

    def get_stop_sequences(self):
        return []

    def parse_response(self, tokens):
        return {"role": "assistant", "content": tokens}, True


class FakeTrainingClient:
    def __init__(self, calls):
        self.calls = calls

    def get_tokenizer(self):
        return "tokenizer"

    def forward_backward(self, data, loss_fn):
        self.calls.append(("forward_backward", len(data), loss_fn))
        return _Done(types.SimpleNamespace(metrics={"loss:sum": 1.5}))

    def optim_step(self, params):
        self.calls.append(("optim_step", params.learning_rate))
        return _Done(None)

    def save_weights_for_sampler(self, name):
        self.calls.append(("save", name))
        return _Done(types.SimpleNamespace(path=f"tinker://run/sampler_weights/{name}"))


class FakeSamplingClient:
    def __init__(self, knows_assamese):
        self.knows_assamese = knows_assamese

    def sample(self, prompt, num_samples, sampling_params):
        plant = next((p for p in PLANTS if p.roman in prompt or p.assamese in prompt or p.english in prompt), None)
        if self.knows_assamese and plant:
            text = f"{plant.assamese} — {plant.roman} ({plant.scientific})"
        else:
            text = "<think>hmm</think>Some kind of fruit tree."
        return _Done(types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=text)]))


@pytest.fixture
def fake_tinker(monkeypatch):
    calls = []

    class ServiceClient:
        def create_lora_training_client(self, base_model, rank):
            calls.append(("lora", base_model, rank))
            return FakeTrainingClient(calls)

        def create_sampling_client(self, base_model=None, model_path=None):
            return FakeSamplingClient(knows_assamese=model_path is not None)

    tinker = types.ModuleType("tinker")
    tinker.ServiceClient = ServiceClient
    tinker.types = types.SimpleNamespace(
        AdamParams=lambda learning_rate: types.SimpleNamespace(learning_rate=learning_rate),
        SamplingParams=lambda **kw: types.SimpleNamespace(**kw),
    )
    renderers = types.ModuleType("tinker_cookbook.renderers")
    renderers.get_renderer = lambda name, tokenizer: FakeRenderer()
    renderers.TrainOnWhat = types.SimpleNamespace(LAST_ASSISTANT_MESSAGE="last")
    data = types.ModuleType("tinker_cookbook.supervised.data")
    data.conversation_to_datum = lambda conv, renderer, max_length, train_on_what: ("datum", conv[-1]["content"])
    weights = types.ModuleType("tinker_cookbook.weights")
    weights.download = lambda tinker_path, output_dir: calls.append(("download", tinker_path)) or output_dir
    weights.build_lora_adapter = lambda **kw: calls.append(("peft", kw["output_path"]))
    weights.build_hf_model = lambda **kw: calls.append(("merge", kw["output_path"]))
    tokenizer_utils = types.ModuleType("tinker_cookbook.tokenizer_utils")
    tokenizer_utils.get_tokenizer = lambda model: calls.append(("tokenizer", model)) or "tokenizer"
    cookbook = types.ModuleType("tinker_cookbook")
    cookbook.weights = weights

    modules = {
        "tinker": tinker,
        "tinker_cookbook": cookbook,
        "tinker_cookbook.tokenizer_utils": tokenizer_utils,
        "tinker_cookbook.renderers": renderers,
        "tinker_cookbook.supervised": types.ModuleType("tinker_cookbook.supervised"),
        "tinker_cookbook.supervised.data": data,
        "tinker_cookbook.weights": weights,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def _write_run(path: Path) -> Path:
    path.write_text(
        json.dumps({"tinker_path": "tinker://run/sampler_weights/x", "base_model": "Qwen/Qwen3.5-4B", "renderer": "qwen3_5"}),
        encoding="utf-8",
    )
    return path


def test_train_runs_every_batch_and_writes_the_run_record(fake_tinker, tmp_path):
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, build_training_set(PLANTS)[:40])
    args = train_tinker.parse_args(
        ["--train", str(train_file), "--epochs", "2", "--batch-size", "16", "--run-file", str(tmp_path / "run.json")]
    )

    record = train_tinker.train(args)

    assert sum(1 for c in fake_tinker if c[0] == "forward_backward") == 6  # ceil(40/16) * 2 epochs
    assert ("lora", "Qwen/Qwen3.5-4B", 32) in fake_tinker
    assert record["tinker_path"] == "tinker://run/sampler_weights/aitar-kotha"
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))["steps"] == 6


def test_evaluate_scores_base_against_tuned_and_writes_reports(fake_tinker, tmp_path):
    eval_file = tmp_path / "eval.jsonl"
    write_jsonl(eval_file, build_eval_set(PLANTS)[:8])
    args = evaluate.parse_args(
        ["--run-file", str(_write_run(tmp_path / "run.json")), "--eval", str(eval_file), "--out-dir", str(tmp_path / "reports")]
    )

    result = evaluate.evaluate(args)

    assert result["tuned"]["assamese"] > result["base"]["assamese"]
    report = (tmp_path / "reports" / "eval.md").read_text(encoding="utf-8")
    assert "Aita (fine-tuned)" in report
    rows = json.loads((tmp_path / "reports" / "eval.json").read_text(encoding="utf-8"))["rows"]
    assert "<think>" not in rows[0]["base"]


def test_export_builds_adapter_and_optionally_merges(fake_tinker, tmp_path):
    args = export_local.parse_args(
        ["--run-file", str(_write_run(tmp_path / "run.json")), "--out-dir", str(tmp_path / "models"), "--merge"]
    )

    export_local.export(args)

    kinds = [c[0] for c in fake_tinker]
    assert kinds == ["download", "peft", "merge"]
    assert "FROM ./aita.gguf" in (tmp_path / "models" / "Modelfile").read_text(encoding="utf-8")


def test_dataset_cli_writes_both_files(tmp_path, capsys):
    from aita import dataset

    dataset.main(["--out-dir", str(tmp_path)])

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()
    assert "training chats" in capsys.readouterr().out
