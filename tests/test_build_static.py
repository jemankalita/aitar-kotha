import json

import pytest

from aita.build_static import build_site
from aita.knowledge import load_plants
from aita.voice import VoiceConfig, VoiceError
from aita.voice_clips import clip_name, generate_clips

PLANTS = load_plants()
CONFIG = VoiceConfig(api_key="secret", voice_id="v", model_id="eleven_v4")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- the static demo site ----------


def test_build_copies_the_web_app(tmp_path):
    out = build_site(tmp_path / "site")

    for name in ("index.html", "app.js", "i18n.js", "achievements.js", "styles.css", "sw.js"):
        assert (out / name).is_file(), name
    assert (out / "fonts" / "noto-serif-bengali.woff2").is_file()


def test_build_writes_plants_with_assamese_notes(tmp_path):
    out = build_site(tmp_path / "site")

    plants = _read_json(out / "plants.json")
    assert len(plants) == len(PLANTS)
    bogori = next(p for p in plants if p["id"] == "bogori")
    assert bogori["assamese"] == "বগৰী"
    assert "সৰস্বতী" in bogori["as"]["story"]


def test_build_turns_on_demo_mode(tmp_path):
    out = build_site(tmp_path / "site", repo_url="https://github.com/x/y")

    config = _read_json(out / "config.json")
    assert config == {"mode": "demo", "repo": "https://github.com/x/y", "voice": False}


def test_build_ships_voice_clips_when_present(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "bogori-as.mp3").write_bytes(b"ID3")

    out = build_site(tmp_path / "site", clips_dir=clips)

    assert (out / "audio" / "bogori-as.mp3").read_bytes() == b"ID3"
    assert _read_json(out / "config.json")["voice"] is True


def test_build_replaces_an_old_site(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="utf-8")

    build_site(out)

    assert not (out / "stale.txt").exists()


def test_build_refuses_to_wipe_the_web_source(tmp_path):
    from aita.build_static import WEB_DIR

    with pytest.raises(ValueError):
        build_site(WEB_DIR)


# ---------- pre-generated voice clips ----------


def test_clip_name_is_predictable():
    assert clip_name("bogori", "as") == "bogori-as.mp3"


def test_generate_clips_makes_one_per_plant_and_language(tmp_path):
    calls = []

    def fake_speak(config, text, lang, **kwargs):
        calls.append(lang)
        return b"ID3" + lang.encode()

    made = generate_clips(PLANTS[:2], {}, CONFIG, tmp_path, speak_fn=fake_speak)

    assert made == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        clip_name(p.id, lang) for p in PLANTS[:2] for lang in ("as", "en")
    )


def test_generate_clips_skips_existing_files(tmp_path):
    (tmp_path / clip_name(PLANTS[0].id, "as")).write_bytes(b"old")

    made = generate_clips(PLANTS[:1], {}, CONFIG, tmp_path, speak_fn=lambda *a, **k: b"new")

    assert made == 1
    assert (tmp_path / clip_name(PLANTS[0].id, "as")).read_bytes() == b"old"


def test_generate_clips_stops_on_voice_error(tmp_path):
    def broken(*args, **kwargs):
        raise VoiceError("ElevenLabs returned HTTP 401")

    with pytest.raises(VoiceError):
        generate_clips(PLANTS[:1], {}, CONFIG, tmp_path, speak_fn=broken)
