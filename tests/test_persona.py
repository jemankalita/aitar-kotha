from aita.knowledge import load_plants, plant_by_id
from aita.persona import (
    SYSTEM_PROMPT,
    build_messages,
    compose_answer,
    facts_block,
    season_answer,
    unknown_answer,
)

PLANTS = load_plants()
BOGORI = plant_by_id(PLANTS, "bogori")


def test_answer_leads_with_assamese_name_then_english_and_scientific():
    answer = compose_answer(BOGORI)

    first_line = answer.splitlines()[0]
    assert first_line.startswith("বগৰী")
    assert "bogori" in first_line
    assert "Indian jujube" in first_line
    assert "Ziziphus mauritiana" in first_line


def test_answer_includes_story_and_caution():
    answer = compose_answer(BOGORI)

    assert "Saraswati Puja" in answer
    assert "thorns" in answer


def test_answer_ends_by_sending_you_outside():
    for variant in range(3):
        last_line = compose_answer(BOGORI, variant=variant).splitlines()[-1].lower()
        assert "phone" in last_line or "go" in last_line


def test_variants_differ_but_keep_the_facts():
    answers = {compose_answer(BOGORI, variant=v) for v in range(3)}

    assert len(answers) == 3
    assert all("Ziziphus mauritiana" in a for a in answers)


def test_answer_stays_short_enough_for_the_trail():
    for plant in PLANTS:
        assert len(compose_answer(plant).split()) <= 170, plant.id


def test_facts_block_lists_the_key_fields():
    block = facts_block(BOGORI)

    for text in ("বগৰী", "bogori", "Indian jujube", "Ziziphus mauritiana", "Saraswati Puja"):
        assert text in block


def test_build_messages_without_context_has_system_and_user():
    messages = build_messages("what is bogori?", ())

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "what is bogori?"


def test_build_messages_with_context_appends_field_notes_to_system():
    messages = build_messages("what is this?", (BOGORI,))

    assert "Field notes" in messages[0]["content"]
    assert "Ziziphus mauritiana" in messages[0]["content"]


def test_unknown_answer_admits_not_knowing():
    assert "don't know" in unknown_answer().lower()


def test_season_answer_names_kati_and_in_season_plants():
    answer = season_answer(10, PLANTS)

    assert "Kati" in answer
    assert "শেৱালি" in answer  # sewali flowers in October
    assert "bogori" not in answer  # bogori is a winter fruit
