import json

from aita.dataset import build_eval_set, build_training_set, write_jsonl
from aita.knowledge import load_plants
from aita.scoring import score_answer, summarize

PLANTS = load_plants()


def test_training_set_is_a_few_hundred_chat_examples():
    examples = build_training_set(PLANTS, seed=7)

    assert 300 <= len(examples) <= 900
    for example in examples:
        roles = [m["role"] for m in example["messages"]]
        assert roles == ["system", "user", "assistant"]


def test_training_set_is_deterministic_for_a_seed():
    assert build_training_set(PLANTS, seed=7) == build_training_set(PLANTS, seed=7)


def test_training_set_covers_every_plant():
    examples = build_training_set(PLANTS, seed=7)
    answers = " ".join(e["messages"][-1]["content"] for e in examples)

    for plant in PLANTS:
        assert plant.scientific in answers, plant.id


def test_training_set_mixes_examples_with_and_without_field_notes():
    examples = build_training_set(PLANTS, seed=7)
    with_notes = sum("Field notes" in e["messages"][0]["content"] for e in examples)

    assert 0.2 * len(examples) < with_notes < 0.8 * len(examples)


def test_training_set_teaches_humility_for_unknown_plants():
    examples = build_training_set(PLANTS, seed=7)

    assert any("don't know" in e["messages"][-1]["content"].lower() for e in examples)


def test_eval_questions_never_appear_in_training():
    train_questions = {e["messages"][1]["content"] for e in build_training_set(PLANTS, seed=7)}
    eval_questions = {row["question"] for row in build_eval_set(PLANTS)}

    assert eval_questions
    assert not eval_questions & train_questions


def test_eval_rows_point_at_real_plants():
    ids = {p.id for p in PLANTS}

    for row in build_eval_set(PLANTS):
        assert row["expected_id"] in ids


def test_write_jsonl_round_trips_unicode(tmp_path):
    path = tmp_path / "out" / "rows.jsonl"
    write_jsonl(path, [{"text": "বগৰী"}])

    assert json.loads(path.read_text(encoding="utf-8")) == {"text": "বগৰী"}


def test_score_answer_checks_each_name_form():
    plant = next(p for p in PLANTS if p.id == "bogori")

    score = score_answer("That is বগৰী, bogori, Ziziphus mauritiana.", plant)
    assert score == {"assamese": True, "roman": True, "scientific": True}

    score = score_answer("That is a jujube.", plant)
    assert score == {"assamese": False, "roman": False, "scientific": False}


def test_score_answer_accepts_known_alias_spellings():
    plant = next(p for p in PLANTS if p.id == "sewali")

    assert score_answer("It is xewali!", plant)["roman"] is True


def test_summarize_reports_rates():
    summary = summarize(
        [
            {"assamese": True, "roman": True, "scientific": False},
            {"assamese": False, "roman": True, "scientific": False},
        ]
    )

    assert summary == {"assamese": 0.5, "roman": 1.0, "scientific": 0.0, "n": 2}


def test_summarize_empty_is_zero():
    assert summarize([]) == {"assamese": 0.0, "roman": 0.0, "scientific": 0.0, "n": 0}
