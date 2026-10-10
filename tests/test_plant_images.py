import json

import pytest

from aita.knowledge import load_plants
from aita.plant_images import (
    image_view,
    load_image_credits,
    parse_commons_credit,
    parse_page_image,
    strip_html,
    wiki_title,
)

PLANTS = load_plants()
BOGORI = next(p for p in PLANTS if p.id == "bogori")


def test_strip_html_keeps_plain_text():
    assert strip_html('<a href="//x">Jane&nbsp;Doe</a> (talk)') == "Jane Doe (talk)"


def test_wiki_title_defaults_to_scientific_name_with_overrides():
    assert wiki_title(BOGORI) == "Ziziphus mauritiana"
    assert wiki_title(next(p for p in PLANTS if p.id == "kol")) == "Banana"


def test_parse_page_image_reads_thumbnail_and_file_name():
    response = {
        "query": {
            "pages": {
                "123": {
                    "title": "Ziziphus mauritiana",
                    "pageimage": "Ber_fruit.jpg",
                    "thumbnail": {"source": "https://upload.wikimedia.org/x/800px-Ber_fruit.jpg"},
                }
            }
        }
    }

    assert parse_page_image(response) == ("https://upload.wikimedia.org/x/800px-Ber_fruit.jpg", "Ber_fruit.jpg")


def test_parse_page_image_without_image_is_none():
    assert parse_page_image({"query": {"pages": {"-1": {"missing": ""}}}}) is None


def _commons(meta, url="https://commons.wikimedia.org/wiki/File:Ber.jpg"):
    return {"query": {"pages": {"9": {"imageinfo": [{"descriptionurl": url, "extmetadata": meta}]}}}}


def test_parse_commons_credit_reads_author_and_licence():
    meta = {
        "Artist": {"value": '<a href="//commons">Jane Doe</a>'},
        "LicenseShortName": {"value": "CC BY-SA 4.0"},
        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
    }

    assert parse_commons_credit(_commons(meta)) == {
        "author": "Jane Doe",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "source": "https://commons.wikimedia.org/wiki/File:Ber.jpg",
    }


def test_parse_commons_credit_accepts_public_domain_without_url():
    meta = {"Artist": {"value": "Unknown"}, "LicenseShortName": {"value": "Public domain"}}

    assert parse_commons_credit(_commons(meta))["license"] == "Public domain"


def test_parse_commons_credit_rejects_non_free_or_missing_files():
    meta = {"Artist": {"value": "X"}, "LicenseShortName": {"value": "Fair use"}, "NonFree": {"value": "true"}}

    assert parse_commons_credit(_commons(meta)) is None
    assert parse_commons_credit({"query": {"pages": {"-1": {"missing": ""}}}}) is None


def test_load_image_credits_keeps_known_plants_only(tmp_path):
    path = tmp_path / "plant_images.json"
    path.write_text(
        json.dumps({"bogori": {"file": "bogori.jpg", "author": "A", "license": "CC0", "license_url": "", "source": "s"},
                    "maple": {"file": "maple.jpg"}}),
        encoding="utf-8",
    )

    credits = load_image_credits(PLANTS, path)

    assert set(credits) == {"bogori"}


def test_load_image_credits_missing_file_is_empty(tmp_path):
    assert load_image_credits(PLANTS, tmp_path / "nope.json") == {}


def test_image_view_points_at_the_bundled_file():
    credit = {"file": "bogori.jpg", "author": "A", "license": "CC0", "license_url": "", "source": "s"}

    assert image_view(credit) == {**credit, "src": "images/plants/bogori.jpg"}
    assert image_view(None) is None


def test_bundled_credits_have_a_licence_and_an_existing_file():
    from aita.plant_images import IMAGES_DIR

    credits = load_image_credits(PLANTS)
    if not credits:
        pytest.skip("images not fetched yet")
    for plant_id, credit in credits.items():
        assert credit["license"], plant_id
        assert (IMAGES_DIR / credit["file"]).is_file(), plant_id


def test_server_and_static_site_give_each_plant_its_photo(tmp_path):
    from fastapi.testclient import TestClient

    from aita.build_static import build_site
    from aita.server import create_app

    credit = {"file": "bogori.jpg", "author": "A", "license": "CC0", "license_url": "", "source": "s"}
    client = TestClient(create_app(PLANTS, images={"bogori": credit}))
    plants = client.get("/api/plants").json()
    assert next(p for p in plants if p["id"] == "bogori")["image"]["src"] == "images/plants/bogori.jpg"
    assert next(p for p in plants if p["id"] == "sewali")["image"] is None

    site = json.loads((build_site(tmp_path / "site") / "plants.json").read_text(encoding="utf-8"))
    bundled = load_image_credits(PLANTS)
    for plant in site:
        assert (plant["image"] is not None) == (plant["id"] in bundled)
