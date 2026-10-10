# আইতাৰ কথা · Aitar Kotha

**Plants of Assam, the way Aita (grandmother) told it.** Point your phone at a plant, or just say its name, and a small open-weight model fine-tuned on Assamese family plant knowledge answers like an Assamese grandmother would:

> **বগৰী — bogori** (Indian jujube, *Ziziphus mauritiana*)
> A thorny small tree with small shiny oval leaves… Eaten raw with salt and chilli, or made into sweet-sour bogori achar dried in the winter sun.
> Many families tell children not to eat bogori until after Saraswati Puja…
> **Now put the phone away and go find one.**

Built for the [Hacktoberfest Open-Source AI Challenge, Week 1: Touch Grass](https://dev.to/challenges/hf26).

## Why open

Closed models know "Indian jujube". They rarely know **bogori**, never know that you wait for Saraswati Puja before eating it, and you can't teach them. With open weights you can:

- **Fine-tune on your own culture's knowledge** and own the result: a LoRA adapter trained on [Tinker](https://tinker-docs.thinkingmachines.ai) on top of `Qwen/Qwen3.5-4B`.
- **Run it on your own machine**: export the adapter and serve it with Ollama or llama.cpp; no account and no per-question cost.
- **Keep working with no signal**: the app falls back to the family notes saved on the phone.

## What's inside

| Path | What it does |
|---|---|
| `data/plants.json` | 43 Assamese plants: names in Assamese script and roman letters, English and scientific names, season, how to recognise them, uses, a story, cautions. |
| `aita/dataset.py` | Turns the notes into ~380 chat examples for fine-tuning, plus 172 held-out questions phrased differently. |
| `aita/train_tinker.py` | LoRA fine-tune on Tinker (trains only on Aita's answers). |
| `aita/evaluate.py` | Base model vs fine-tuned Aita on the held-out questions → `reports/eval.md`. |
| `aita/export_local.py` | Downloads the adapter (PEFT) and optionally merges it for Ollama/llama.cpp. |
| `aita/server.py` | FastAPI app: ask by name, by photo (open vision model), or by season. |
| `data/plants_as.json` | The same notes in Assamese (a draft to check with elders). |
| `web/` | Web app for phone and PC: camera, this month's plants, English ⇄ অসমীয়া switch, achievements, a "plants I met" journal, offline fallback. |
| `web/achievements.js` | Badge rules: one badge per plant you scan, five collections, milestones, and আইতাৰ তৰা (Aita's Star) for all 43. |

## Achievements

Scanning a plant with the camera earns its badge (asking by name doesn't count, so you have to actually go and find it). Badges add up to:

- **Milestones:** প্ৰথম খোজ First step (1), পাঁচবিধ (5), দহবিধ (10), বিশবিধ (20)
- **Collections:** টেঙাৰ দল the sour ones, বিহুৰ গছ Bihu plants, পাকঘৰৰ শাক kitchen greens, চোতালৰ ফল courtyard fruits, হাবি আৰু শিল্প forest and craft
- **আইতাৰ তৰা Aita's Star:** every one of the 43 plants

Progress is stored only in that browser (localStorage) and never leaves the device.

## 1. Set up

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
pytest                            # Python tests
node --test tests/js/*.test.cjs   # badge rules
```

## 2. Fine-tune Aita on Tinker

```bash
pip install -e ".[train]"         # tinker + tinker-cookbook (use Python 3.11-3.13 if 3.14 wheels are missing)
export TINKER_API_KEY=...         # PowerShell: $env:TINKER_API_KEY="..."
python -m aita.dataset            # data/sft/train.jsonl + eval.jsonl
python -m aita.train_tinker       # ~96 steps; writes runs/latest.json
python -m aita.evaluate           # reports/eval.md: base vs Aita
```

Defaults: `Qwen/Qwen3.5-4B`, LoRA rank 32, learning rate 2e-4 decaying linearly, 4 epochs, batch 16. The run is about 1M training tokens. Check current pricing on the Tinker models page.

## 3. Run the app

Without any model, the app answers from the family notes:

```bash
uvicorn aita.server:build_default_app --factory --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` (or `http://<your-laptop-ip>:8000` on your phone, on the same Wi-Fi).

**Use fine-tuned Aita straight from Tinker** (OpenAI-compatible endpoint):

```bash
export AITA_STORY_BASE_URL=https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
export AITA_STORY_MODEL="$(python -c "import json;print(json.load(open('runs/latest.json'))['tinker_path'])")"
export AITA_STORY_API_KEY=$TINKER_API_KEY
```

**Or run Aita fully locally** with Ollama:

```bash
python -m aita.export_local --merge
python path/to/llama.cpp/convert_hf_to_gguf.py models/merged --outfile models/aita.gguf --outtype q8_0
ollama create aita -f models/Modelfile
export AITA_STORY_BASE_URL=http://localhost:11434/v1
export AITA_STORY_MODEL=aita
```

**Photo identification** uses an open vision model on Ollama (Gemma 4 or any vision model you have):

```bash
ollama pull gemma4:e4b            # use whichever Gemma vision tag `ollama list` / the Ollama library shows
export AITA_VISION_BASE_URL=http://localhost:11434/v1
export AITA_VISION_MODEL=gemma4:e4b
```

The vision model only picks from the 43 plants in the notes and reports a confidence. Below 50% Aita says she isn't sure instead of guessing.

**Aita's voice in Assamese (optional, ElevenLabs).** Every card gets a 🔊 শুনা · Listen button that reads the name, Aita's story and any warning aloud, in Assamese or English (`eleven_v4`, `language_code="as"`). Copy `.env.example` to `.env` and set `ELEVENLABS_API_KEY` (create one at https://elevenlabs.io/app/settings/api-keys; never commit `.env`). Each clip is generated once and cached in `audio_cache/`, so replays cost nothing and work offline. Change `AITA_VOICE_ID` to whichever voice sounds most like your Aita. This is the one closed service in the project; without a key the button simply doesn't appear.

## Offline on the trail

The service worker saves the app and the plant notes after the first visit, so asking by name keeps working without signal. Browsers only enable service workers on `localhost` or HTTPS: on a phone, serve over HTTPS (for example with `mkcert` and `uvicorn --ssl-keyfile/--ssl-certfile`) for offline mode to work. Photo identification needs the laptop running the vision model to be reachable.

## Honesty notes

- The Assamese text (`data/plants_as.json` and the Assamese parts of `web/i18n.js`) was drafted with AI help and **must be checked by a fluent speaker** before you rely on it.
- The notes in `data/plants.json` are a **first draft that still needs checking with elders**. Every entry has `"verified": false` until someone in the family confirms it, and the app shows that.
- Traditional remedies are recorded as culture, **not medical advice**.
- Aita is trained to say "I don't know" for plants outside the notes and never to tell anyone to eat something they cannot name for sure.

## Add your family's plants

Add an entry to `data/plants.json` (the loader validates every field), set `"verified": true` once an elder has checked it, then rebuild the dataset and fine-tune again.

## License

MIT. See `LICENSE`.
