import json

import pytest

from aita.knowledge import (
    KnowledgeError,
    assamese_month,
    in_season,
    load_plants,
    parse_plant,
    plant_by_id,
)


def _raw(**overrides):
    base = {
        "id": "bogori",
        "assamese": "বগৰী",
        "roman": "bogori",
        "aliases": ["ber", "Indian Jujube"],
        "english": "Indian jujube",
        "scientific": "Ziziphus mauritiana",
        "kind": "tree",
        "months": [12, 1],
        "season_note": "Winter.",
        "look_for": "Thorny tree.",
        "uses": ["Eaten raw."],
        "story": "Wait for Saraswati Puja.",
        "caution": None,
        "verified": False,
    }
    return {**base, **overrides}


def test_parse_plant_builds_immutable_record():
    plant = parse_plant(_raw())

    assert plant.id == "bogori"
    assert plant.months == (12, 1)
    assert plant.uses == ("Eaten raw.",)
    with pytest.raises(AttributeError):
        plant.id = "other"  # frozen dataclass


def test_parse_plant_lowercases_aliases():
    plant = parse_plant(_raw())

    assert "indian jujube" in plant.aliases


def test_names_include_every_way_to_say_it():
    plant = parse_plant(_raw())

    assert {"bogori", "বগৰী", "indian jujube", "ziziphus mauritiana", "ber"} <= set(plant.names())


@pytest.mark.parametrize("field", ["id", "assamese", "roman", "english", "story"])
def test_parse_plant_rejects_missing_text(field):
    with pytest.raises(KnowledgeError, match=field):
        parse_plant(_raw(**{field: ""}))


def test_parse_plant_rejects_bad_month():
    with pytest.raises(KnowledgeError, match="months"):
        parse_plant(_raw(months=[13]))


def test_load_plants_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "plants.json"
    path.write_text(json.dumps([_raw(), _raw()]), encoding="utf-8")

    with pytest.raises(KnowledgeError, match="duplicate"):
        load_plants(path)


def test_bundled_knowledge_base_loads_and_has_bogori():
    plants = load_plants()

    bogori = plant_by_id(plants, "bogori")
    assert bogori.english == "Indian jujube"
    assert bogori.scientific == "Ziziphus mauritiana"
    assert len(plants) >= 40


def test_plant_by_id_unknown_raises():
    with pytest.raises(KeyError):
        plant_by_id(load_plants(), "maple")


def test_in_season_filters_by_month():
    plants = (parse_plant(_raw()), parse_plant(_raw(id="jamu", months=[6])))

    assert [p.id for p in in_season(plants, 1)] == ["bogori"]
    assert [p.id for p in in_season(plants, 6)] == ["jamu"]


def test_in_season_rejects_bad_month():
    with pytest.raises(ValueError):
        in_season((), 0)


def test_assamese_month_for_october_is_kati():
    assert assamese_month(10) == "Kati"
    assert assamese_month(4) == "Bohag"


# ---------- Assamese translations ----------

from aita.knowledge import load_translations  # noqa: E402


def _translation(**overrides):
    base = {
        "season_note": "জাৰৰ দিনত।",
        "look_for": "কাঁইটীয়া সৰু গছ।",
        "uses": ["কেঁচাকৈ খোৱা হয়।"],
        "story": "সৰস্বতী পূজাৰ আগতে নাখাবা।",
        "caution": None,
    }
    return {**base, **overrides}


def _write_translations(tmp_path, plants_map):
    path = tmp_path / "plants_as.json"
    path.write_text(json.dumps({"verified": False, "plants": plants_map}, ensure_ascii=False), encoding="utf-8")
    return path


def test_bundled_translations_cover_every_plant_in_assamese():
    plants = load_plants()
    translations = load_translations(plants=plants)

    assert set(translations) == {p.id for p in plants}
    assert "সৰস্বতী" in translations["bogori"]["story"]


def test_load_translations_rejects_unknown_plant_ids(tmp_path):
    path = _write_translations(tmp_path, {"maple": _translation()})

    with pytest.raises(KnowledgeError, match="maple"):
        load_translations(path, plants=load_plants())


def test_load_translations_rejects_missing_fields(tmp_path):
    path = _write_translations(tmp_path, {"bogori": _translation(story="")})

    with pytest.raises(KnowledgeError, match="story"):
        load_translations(path, plants=load_plants())


def test_load_translations_missing_file_means_no_translations(tmp_path):
    assert load_translations(tmp_path / "nope.json", plants=load_plants()) == {}
