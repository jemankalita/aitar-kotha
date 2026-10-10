from datetime import datetime, timezone

import pytest

from aita.evaluate import comparison_table, message_text
from aita.train_tinker import epoch_batches, learning_rate, run_record


def test_epoch_batches_cover_every_item_once_per_epoch():
    batches = epoch_batches(n_items=10, batch_size=4, epochs=2, seed=1)

    assert [len(b) for b in batches] == [4, 4, 2, 4, 4, 2]
    first_epoch = sorted(i for b in batches[:3] for i in b)
    assert first_epoch == list(range(10))


def test_epoch_batches_shuffle_differently_each_epoch_but_deterministically():
    batches = epoch_batches(n_items=20, batch_size=20, epochs=2, seed=1)

    assert batches[0] != batches[1]
    assert batches == epoch_batches(n_items=20, batch_size=20, epochs=2, seed=1)


@pytest.mark.parametrize("bad", [{"n_items": 0}, {"batch_size": 0}, {"epochs": 0}])
def test_epoch_batches_reject_empty_inputs(bad):
    args = {"n_items": 5, "batch_size": 2, "epochs": 1, "seed": 0, **bad}

    with pytest.raises(ValueError):
        epoch_batches(**args)


def test_learning_rate_decays_linearly_to_a_floor():
    assert learning_rate(0, 10, 1e-4) == pytest.approx(1e-4)
    assert learning_rate(5, 10, 1e-4) == pytest.approx(0.55e-4)
    assert learning_rate(10, 10, 1e-4) == pytest.approx(1e-5)


def test_run_record_has_what_the_app_needs():
    now = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)

    record = run_record("tinker://abc/sampler_weights/final", "Qwen/Qwen3.5-4B", "qwen3_5", 72, now)

    assert record == {
        "tinker_path": "tinker://abc/sampler_weights/final",
        "base_model": "Qwen/Qwen3.5-4B",
        "renderer": "qwen3_5",
        "steps": 72,
        "created_at": "2026-10-07T12:00:00+00:00",
    }


def test_message_text_handles_strings_and_parts():
    assert message_text({"content": " hi "}) == "hi"
    assert message_text({"content": [{"type": "thinking", "text": "x"}, {"type": "text", "text": "বগৰী"}]}) == "বগৰী"
    assert message_text({}) == ""


def test_comparison_table_shows_base_and_tuned_rates():
    table = comparison_table(
        {"assamese": 0.05, "roman": 0.4, "scientific": 0.6, "n": 172},
        {"assamese": 0.9, "roman": 0.95, "scientific": 0.97, "n": 172},
    )

    assert "| Assamese script name | 5% | 90% |" in table
    assert "172" in table
