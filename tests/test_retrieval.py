import pytest

from aita.knowledge import load_plants
from aita.retrieval import find_plants, normalize

PLANTS = load_plants()


def _ids(question):
    return [p.id for p in find_plants(question, PLANTS)]


def test_normalize_strips_punctuation_and_case():
    assert normalize("  Aita, what's BOGORI?! ") == "aita what s bogori"


def test_normalize_keeps_assamese_script():
    assert normalize("বগৰী কি?") == "বগৰী কি"


@pytest.mark.parametrize(
    "question",
    [
        "Aita, what is bogori?",
        "What is Indian jujube called in Assamese?",
        "Tell me about Ziziphus mauritiana",
        "বগৰী কি?",
        "I picked some ber today",
    ],
)
def test_finds_bogori_however_it_is_named(question):
    assert _ids(question)[0] == "bogori"


def test_tolerates_small_spelling_mistakes():
    assert _ids("what is bogri")[0] == "bogori"


def test_prefers_longest_name_match():
    # "tenga" alone is not a plant, but "ou tenga" is.
    assert _ids("cook fish with ou tenga")[0] == "ou_tenga"


def test_does_not_match_inside_other_words():
    # "ber" must not match inside "number" or "remember".
    assert "bogori" not in _ids("remember the number")


def test_returns_nothing_for_unknown_plants():
    assert _ids("what is a saguaro cactus") == []


def test_limit_caps_results():
    found = find_plants("tamul and paan and bogori", PLANTS, limit=2)

    assert len(found) == 2
