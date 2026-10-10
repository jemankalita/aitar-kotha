import pytest
from fastapi.testclient import TestClient

from aita.knowledge import load_plants
from aita.llm import Endpoint, Identification, LLMError
from aita.server import MAX_IMAGE_BYTES, create_app

PLANTS = load_plants()
STORY = Endpoint(base_url="http://story/v1", model="aita", api_key=None)
JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"0" * 32
VISION = Endpoint(base_url="http://vision/v1", model="gemma", api_key=None)


def _app(story=None, vision=None, chat_fn=None, identify_fn=None):
    return TestClient(
        create_app(
            PLANTS,
            story=story,
            vision=vision,
            chat_fn=chat_fn or (lambda *a, **k: "unused"),
            identify_fn=identify_fn or (lambda *a, **k: Identification(None, 0.0, "")),
        )
    )


def test_health_reports_which_models_are_wired():
    body = _app(story=STORY).get("/api/health").json()

    assert body == {"story": True, "vision": False, "voice": False, "plants": len(PLANTS)}


def test_plants_endpoint_lists_the_knowledge_base():
    body = _app().get("/api/plants").json()

    assert any(p["id"] == "bogori" and p["assamese"] == "বগৰী" for p in body)


def test_season_endpoint_returns_kati_plants():
    body = _app().get("/api/season", params={"month": 10}).json()

    assert body["assamese_month"] == "Kati"
    assert "sewali" in [p["id"] for p in body["plants"]]


def test_season_endpoint_validates_month():
    assert _app().get("/api/season", params={"month": 13}).status_code == 422


def test_ask_uses_the_model_with_field_notes_when_available():
    seen = {}

    def fake_chat(endpoint, messages, **kwargs):
        seen["messages"] = messages
        return "বগৰী! Wait for Saraswati Puja, child."

    body = _app(story=STORY, chat_fn=fake_chat).post("/api/ask", json={"question": "what is bogori?"}).json()

    assert body["source"] == "model"
    assert body["plant"]["id"] == "bogori"
    assert "Ziziphus mauritiana" in seen["messages"][0]["content"]


def test_ask_falls_back_to_family_notes_without_a_model():
    body = _app().post("/api/ask", json={"question": "what is bogori?"}).json()

    assert body["source"] == "notes"
    assert body["answer"].startswith("বগৰী")


def test_ask_falls_back_to_notes_when_the_model_fails():
    def broken_chat(*args, **kwargs):
        raise LLMError("down")

    body = _app(story=STORY, chat_fn=broken_chat).post("/api/ask", json={"question": "bogori?"}).json()

    assert body["source"] == "notes"


def test_ask_unknown_plant_without_model_admits_it():
    body = _app().post("/api/ask", json={"question": "what is a saguaro cactus?"}).json()

    assert body["plant"] is None
    assert "don't know" in body["answer"].lower()


def test_ask_season_question_lists_this_month():
    body = _app().post("/api/ask", json={"question": "what can I find this month?", "month": 10}).json()

    assert "Kati" in body["answer"]


@pytest.mark.parametrize("question", ["", " ", "x" * 501])
def test_ask_validates_question(question):
    assert _app().post("/api/ask", json={"question": question}).status_code == 422


def test_identify_without_vision_model_is_503():
    response = _app().post("/api/identify", files={"image": ("leaf.jpg", JPEG, "image/jpeg")})

    assert response.status_code == 503


def test_identify_rejects_non_images():
    response = _app(vision=VISION).post("/api/identify", files={"image": ("a.txt", b"hi", "text/plain")})

    assert response.status_code == 415


def test_identify_rejects_huge_images():
    big = b"0" * (MAX_IMAGE_BYTES + 1)
    response = _app(vision=VISION).post("/api/identify", files={"image": ("a.jpg", big, "image/jpeg")})

    assert response.status_code == 413


def test_identify_returns_plant_and_aita_answer():
    def fake_identify(endpoint, data, mime, plants, **kwargs):
        return Identification("sewali", 0.9, "orange tubes")

    body = _app(vision=VISION, identify_fn=fake_identify).post(
        "/api/identify", files={"image": ("leaf.jpg", JPEG, "image/jpeg")}
    ).json()

    assert body["plant"]["id"] == "sewali"
    assert body["confidence"] == 0.9
    assert body["answer"].startswith("শেৱালি")


def test_identify_low_confidence_says_aita_is_unsure():
    def unsure(*args, **kwargs):
        return Identification("sewali", 0.2, "blurry")

    body = _app(vision=VISION, identify_fn=unsure).post(
        "/api/identify", files={"image": ("leaf.jpg", JPEG, "image/jpeg")}
    ).json()

    assert body["plant"] is None
    assert body["guess"]["id"] == "sewali"


def test_identify_vision_failure_is_502():
    def broken(*args, **kwargs):
        raise LLMError("down")

    response = _app(vision=VISION, identify_fn=broken).post(
        "/api/identify", files={"image": ("leaf.jpg", JPEG, "image/jpeg")}
    )

    assert response.status_code == 502


def test_home_page_is_served():
    response = _app().get("/")

    assert response.status_code == 200
    assert "Aitar Kotha" in response.text


def test_identify_rejects_files_that_only_claim_to_be_images():
    response = _app(vision=VISION).post("/api/identify", files={"image": ("x.jpg", b"<script>", "image/jpeg")})

    assert response.status_code == 415


def test_identify_passes_the_sniffed_type_to_the_vision_model():
    seen = {}

    def fake_identify(endpoint, data, mime, plants, **kwargs):
        seen["mime"] = mime
        return Identification(None, 0.0, "")

    png = b"\x89PNG\r\n\x1a\n" + b"0" * 16
    _app(vision=VISION, identify_fn=fake_identify).post(
        "/api/identify", files={"image": ("leaf.jpg", png, "image/png;evil,x")}
    )

    assert seen["mime"] == "image/png"


def test_plants_carry_their_assamese_notes():
    client = TestClient(create_app(PLANTS, translations={"bogori": {"story": "সৰস্বতী পূজা"}}))

    bogori = next(p for p in client.get("/api/plants").json() if p["id"] == "bogori")
    assert bogori["as"] == {"story": "সৰস্বতী পূজা"}


def test_plants_without_translation_have_null_assamese():
    sewali = next(p for p in _app().get("/api/plants").json() if p["id"] == "sewali")

    assert sewali["as"] is None


@pytest.mark.parametrize(
    ("question", "kind"),
    [("what is bogori?", "plant"), ("what can I find this month?", "season"), ("what is a saguaro?", "unknown")],
)
def test_ask_says_what_kind_of_answer_it_is(question, kind):
    assert _app().post("/api/ask", json={"question": question, "month": 10}).json()["kind"] == kind
