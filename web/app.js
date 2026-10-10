"use strict";

// Aitar Kotha front end: ask by name or photo, read in English or Assamese, earn badges by scanning.
// Talks to the local server; falls back to the family notes saved on the phone when offline.

const { STRINGS, LABELS, KINDS } = self.AitaI18n;
const A = self.AitaAchievements;

const JOURNAL_KEY = "aita.met";
const BADGE_KEY = "aita.badges";
const LANG_KEY = "aita.lang";
const MAX_PHOTO_SIDE = 1280;
const PHOTO_QUALITY = 0.85;
const MONTH = new Date().getMonth() + 1;
const MONTHS = [
  ["মাঘ", "Magh"], ["ফাগুন", "Phagun"], ["চ'ত", "Sot"], ["ব'হাগ", "Bohag"],
  ["জেঠ", "Jeth"], ["আহাৰ", "Ahar"], ["শাওণ", "Xaon"], ["ভাদ", "Bhado"],
  ["আহিন", "Ahin"], ["কাতি", "Kati"], ["আঘোণ", "Aghon"], ["পুহ", "Puh"],
];
const KIND_ICONS = {
  tree: "i-tree", palm: "i-tree", shrub: "i-leaf", herb: "i-leaf",
  orchid: "i-flower", aquatic: "i-flower", climber: "i-vine", grass: "i-grass", fern: "i-grass",
};
const SET_ICONS = { tenga: "i-drop", bihu: "i-flower", xaak: "i-leaf", fol: "i-fruit", habi: "i-tree" };
const SOURCE_KEYS = { model: "sourceModel", notes: "sourceNotes", offline: "sourceOffline" };

const $ = (id) => document.getElementById(id);
let plants = [];
let currentPlant = null;
let lastResult = null;
let lang = readStored(LANG_KEY) === "as" ? "as" : "en";
let badges = A.parseState(readStored(BADGE_KEY) || "");
let voiceOn = false; // set from /api/health: true when the server has an ElevenLabs key
let voiceAudio = null;

// ---------- small helpers ----------

function readStored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    setStatus(t("saveFail"));
  }
}

function t(key, vars = {}) {
  const text = STRINGS[lang][key] ?? STRINGS.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
}

const isAs = () => lang === "as";

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children);
  return node;
}

function icon(id) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${id}`);
  svg.append(use);
  return svg;
}

function assamese(text) {
  return el("span", { lang: "as", textContent: text });
}

function bilingual([as, en]) {
  return isAs() ? [assamese(as)] : [assamese(as), en];
}

function notesOf(plant) {
  return isAs() && plant.as ? plant.as : plant;
}

function awardName(award) {
  return isAs() ? award.as : award.en;
}

function localDate(date = new Date()) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function normalize(text) {
  return ` ${text.toLowerCase().replace(/[^a-z0-9ঀ-৿]+/g, " ").trim()} `;
}

function findPlantOffline(question) {
  const q = normalize(question);
  let best = null;
  let bestLength = 0;
  for (const plant of plants) {
    for (const name of [plant.roman, plant.assamese, plant.english, plant.scientific, ...plant.aliases]) {
      const key = normalize(name);
      if (key.trim() && q.includes(key) && key.length > bestLength) {
        best = plant;
        bestLength = key.length;
      }
    }
  }
  return best;
}

function setStatus(text, loading = false) {
  $("status").textContent = text;
  $("status").classList.toggle("is-loading", loading);
}

async function errorText(response) {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : `Error ${response.status}`;
  } catch {
    return `Error ${response.status}`;
  }
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

// ---------- language ----------

function applyStaticText() {
  document.documentElement.lang = isAs() ? "as" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => (node.textContent = t(node.dataset.i18n)));
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => (node.placeholder = t(node.dataset.i18nPlaceholder)));
  document.querySelectorAll("[data-i18n-label]").forEach((node) => node.setAttribute("aria-label", t(node.dataset.i18nLabel)));
  $("lang-label").textContent = t("langButton");
  $("lang-label").lang = isAs() ? "en" : "as";
  $("lang-toggle").setAttribute("aria-pressed", String(isAs()));
  $("card-lang").textContent = t("readOther");
}

function setLanguage(next) {
  stopVoice();
  lang = next;
  store(LANG_KEY, lang);
  applyStaticText();
  renderSeason();
  renderJournal();
  renderAchievements();
  if (lastResult) renderAnswer(lastResult, { scroll: false });
}

// ---------- the field card ----------

function fact(key, text, className = "") {
  return el("div", { className }, el("dt", {}, ...bilingual(LABELS[key])), el("dd", { textContent: text }));
}

function renderHead(plant) {
  $("card-head").hidden = !plant;
  if (!plant) return;
  const [kindAs, kindEn] = KINDS[plant.kind] || ["", plant.kind];
  $("kind").replaceChildren(icon(KIND_ICONS[plant.kind] || "i-leaf"), isAs() ? assamese(kindAs) : kindEn);
  $("as-name").textContent = plant.assamese;
  $("roman").textContent = plant.roman;
  $("english").textContent = plant.english;
  $("scientific").textContent = plant.scientific;
  const earned = Boolean(badges.scans[plant.id]);
  $("badge-chip").hidden = !earned;
  $("badge-chip").replaceChildren(icon("i-star"), t("badgeEarned"));
  $("scan-hint").hidden = earned;
  $("scan-hint").textContent = t("scanToEarn");
}

function stripHeader(lines, plant) {
  const isHeader = plant && lines[0]?.startsWith(plant.assamese) && lines[0].includes(plant.scientific);
  return isHeader ? lines.slice(1) : lines;
}

function seasonLinesAs() {
  const list = plants.filter((p) => p.months.includes(MONTH));
  return [t("seasonHead", { month: MONTHS[MONTH - 1][0] }), ...list.map((p) => `• ${p.assamese}`), t("seasonTail")];
}

function quoteFor(result) {
  const { plant, source, kind } = result;
  if (plant && source !== "model") return [notesOf(plant).story];
  if (source === "model" || !isAs()) return stripHeader(result.answer.split("\n").filter((l) => l.trim()), plant);
  if (kind === "season") return seasonLinesAs();
  if (kind === "unknown") return [t("unknown")];
  return [result.guess ? t("unsureGuess", { name: result.guess.assamese }) : t("unsure")];
}

function renderFacts(plant, fromModel) {
  const facts = [];
  if (plant) {
    const notes = notesOf(plant);
    facts.push(fact("look", notes.look_for));
    if (!fromModel) facts.push(fact("season", notes.season_note), fact("uses", notes.uses.join(" ")));
    if (notes.caution) facts.push(fact("caution", notes.caution, "caution"));
  }
  $("facts").replaceChildren(...facts);
}

function renderMetButton(plant) {
  const met = $("met");
  met.hidden = !plant;
  if (!plant) return;
  const done = readJournal().some((e) => e.id === plant.id && e.date === localDate());
  met.classList.toggle("is-done", done);
  met.textContent = done ? t("foundDone") : t("found");
}

function renderAnswer(result, { scroll = true } = {}) {
  if (result.plant?.id !== currentPlant?.id) stopVoice();
  lastResult = result;
  const plant = result.plant;
  currentPlant = plant;
  renderHead(plant);
  const lines = quoteFor(result);
  $("aita-words").hidden = lines.length === 0;
  $("aita-label").replaceChildren(...bilingual(["আইতাই কয়", "Aita says"]));
  $("aita-quote").replaceChildren(...lines.map((line) => el("p", { textContent: line })));
  renderFacts(plant, result.source === "model");

  const notes = [t(SOURCE_KEYS[result.source] || "sourceNotes")];
  if (plant && !plant.verified) notes.push(t("unverified"));
  $("badges").textContent = notes.join(" · ");
  renderMetButton(plant);
  renderListenButton(plant);

  const card = $("answer");
  $("welcome").hidden = true;
  card.hidden = false;
  if (!scroll) return;
  card.style.animation = "none";
  void card.offsetWidth; // restart the entrance animation for each new answer
  card.style.animation = "";
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- Aita's voice (ElevenLabs, optional) ----------

function renderListenButton(plant) {
  const button = $("listen");
  button.hidden = !(voiceOn && plant);
  const playing = Boolean(voiceAudio);
  button.textContent = playing ? t("stopListening") : t("listen");
  button.setAttribute("aria-pressed", String(playing));
}

function stopVoice() {
  if (!voiceAudio) return;
  voiceAudio.pause();
  voiceAudio = null;
  renderListenButton(currentPlant);
}

function toggleVoice() {
  if (voiceAudio) {
    stopVoice();
    return;
  }
  if (!currentPlant) return;
  const audio = new Audio(`/api/speak/${encodeURIComponent(currentPlant.id)}?lang=${lang}`);
  voiceAudio = audio;
  renderListenButton(currentPlant);
  setStatus(t("voiceLoading"), true);
  audio.addEventListener("playing", () => setStatus(""));
  audio.addEventListener("ended", stopVoice);
  audio.addEventListener("error", () => {
    stopVoice();
    setStatus(t("voiceFail"));
  });
  audio.play().catch(() => {
    stopVoice();
    setStatus(t("voiceFail"));
  });
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    voiceOn = response.ok && Boolean((await response.json()).voice);
  } catch {
    voiceOn = false; // offline: no voice, everything else still works
  }
  renderListenButton(currentPlant);
}

// ---------- season tray ----------

function renderMonth() {
  const [as, en] = MONTHS[MONTH - 1];
  $("month-as").textContent = as;
  $("month-en").textContent = en;
}

function renderSeason() {
  const list = plants.filter((p) => p.months.includes(MONTH));
  $("season-title").textContent = list.length ? t("seasonCount", { n: list.length }) : t("seasonTitle");
  const items = list.map((plant) => {
    const button = el(
      "button",
      { type: "button", className: "specimen" },
      icon(KIND_ICONS[plant.kind] || "i-leaf"),
      assamese(plant.assamese),
      el("span", { className: "roman", textContent: plant.roman }),
    );
    button.addEventListener("click", () => ask(`Aita, what is ${plant.roman}?`));
    return el("li", {}, button);
  });
  $("season-chips").replaceChildren(...items);
}

// ---------- journal (per-device, stays on the phone) ----------

function readJournal() {
  try {
    const entries = JSON.parse(readStored(JOURNAL_KEY) || "[]");
    return Array.isArray(entries) ? entries : [];
  } catch {
    return [];
  }
}

function renderJournal() {
  const entries = readJournal();
  const rows = entries.map((entry) =>
    el(
      "li",
      {},
      assamese(entry.assamese),
      el("span", { className: "roman", textContent: entry.roman }),
      el("time", { dateTime: entry.date, textContent: entry.date }),
    ),
  );
  $("met-list").replaceChildren(...rows);
  $("met-count").textContent = String(new Set(entries.map((e) => e.id)).size);
  $("met-empty").hidden = entries.length > 0;
}

function markFound() {
  if (!currentPlant) return;
  const today = localDate();
  const entries = readJournal();
  if (!entries.some((e) => e.id === currentPlant.id && e.date === today)) {
    const entry = { id: currentPlant.id, assamese: currentPlant.assamese, roman: currentPlant.roman, date: today };
    store(JOURNAL_KEY, JSON.stringify([entry, ...entries]));
  }
  renderJournal();
  renderMetButton(currentPlant);
  setStatus(t("journalSaved", { name: isAs() ? currentPlant.assamese : currentPlant.roman }));
}

// ---------- achievements ----------

function medal(content, earned, extra = "") {
  const node = el("span", { className: `medal ${extra} ${earned ? "is-earned" : "is-locked"}` });
  node.setAttribute("aria-hidden", "true");
  node.append(typeof content === "string" ? el("b", { textContent: content }) : content);
  return node;
}

function awardItem(award, earned, detail, content) {
  return el(
    "li",
    { className: `award ${earned ? "is-earned" : ""}` },
    medal(content, earned),
    el("span", { className: "award-text" }, el("strong", { textContent: awardName(award) }), el("small", { textContent: detail })),
  );
}

function renderStar(count, total) {
  const done = total > 0 && count >= total;
  $("star-card").classList.toggle("is-earned", done);
  $("star-name").textContent = awardName(A.STAR);
  $("star-hint").textContent = done ? t("starDone") : t("starHint", { n: total });
  const bar = $("star-bar");
  bar.setAttribute("aria-valuemax", String(total));
  bar.setAttribute("aria-valuenow", String(count));
  bar.setAttribute("aria-label", t("starProgress", { done: count, total }));
  $("star-fill").style.width = total ? `${(count / total) * 100}%` : "0";
}

function renderAwards(count) {
  const milestones = A.MILESTONES.map((m) =>
    awardItem(m, count >= m.need, count >= m.need ? t("complete") : `${Math.min(count, m.need)}/${m.need}`, String(m.need)),
  );
  const sets = A.SETS.map((set) => {
    const { done, total } = A.progress(badges, set.plants);
    return awardItem(set, done === total, `${done}/${total}`, icon(SET_ICONS[set.id]));
  });
  $("award-list").replaceChildren(...milestones, ...sets);
}

function renderPlantBadges() {
  const items = plants.map((plant) => {
    const earned = Boolean(badges.scans[plant.id]);
    const button = el(
      "button",
      { type: "button", className: `plant-badge ${earned ? "is-earned" : "is-locked"}` },
      medal(icon(KIND_ICONS[plant.kind] || "i-leaf"), earned, "medal--small"),
      assamese(plant.assamese),
    );
    button.setAttribute("aria-label", `${plant.assamese} ${plant.roman}: ${earned ? t("badgeEarned") : t("notEarned")}`);
    button.addEventListener("click", () => ask(`Aita, what is ${plant.roman}?`));
    return el("li", {}, button);
  });
  $("plant-badges").replaceChildren(...items);
}

function renderAchievements() {
  const count = plants.filter((p) => badges.scans[p.id]).length;
  $("ach-count").textContent = `${count}/${plants.length}`;
  renderStar(count, plants.length);
  renderAwards(count);
  renderPlantBadges();
}

function celebrate(plant, awards) {
  const dialog = $("celebrate");
  const iconId = awards.includes(A.STAR.id) ? "i-star" : KIND_ICONS[plant.kind] || "i-leaf";
  $("cel-icon").replaceChildren(...icon(iconId).childNodes);
  $("cel-title").replaceChildren(assamese(plant.assamese), el("span", { className: "roman", textContent: ` ${plant.roman}` }));
  $("cel-sub").textContent = t("badgeEarned");
  $("cel-awards").replaceChildren(
    ...awards.map((id) => el("li", {}, icon(SET_ICONS[id] || "i-star"), awardName(A.award(id)))),
  );
  if (typeof dialog.showModal === "function") {
    dialog.showModal();
  } else {
    setStatus(`${t("newBadge")} ${plant.roman}`);
  }
}

function earnBadge(plant) {
  const before = badges;
  const after = A.recordScan(before, plant.id, localDate());
  if (after === before) return;
  badges = after;
  store(BADGE_KEY, JSON.stringify(badges));
  renderAchievements();
  renderHead(plant);
  const { awards } = A.newBadges(before, after, plants.map((p) => p.id));
  celebrate(plant, awards);
}

// ---------- actions ----------

async function ask(question) {
  const text = question.trim();
  if (!text) return;
  setStatus(t("thinking"), true);
  try {
    renderAnswer(await postJSON("/api/ask", { question: text, month: MONTH }));
    setStatus("");
    $("question").value = "";
  } catch {
    const plant = findPlantOffline(text);
    if (plant) {
      renderAnswer({ plant, answer: "", source: "offline", kind: "plant" });
      setStatus(t("offlineNotes"));
    } else {
      setStatus(t("offlineMissing"));
    }
  }
}

// Phone photos are often 8 MB+; a 1280px JPEG is plenty to recognise a leaf and uploads fast.
async function shrinkPhoto(file) {
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, MAX_PHOTO_SIDE / Math.max(bitmap.width, bitmap.height));
    const canvas = el("canvas", { width: Math.round(bitmap.width * scale), height: Math.round(bitmap.height * scale) });
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", PHOTO_QUALITY));
    return blob || file;
  } catch {
    return file; // old browsers: send the original and let the server decide
  }
}

async function identify(file) {
  if (!file) return;
  setStatus(t("looking"), true);
  const form = new FormData();
  form.append("image", await shrinkPhoto(file), "plant.jpg");
  try {
    const response = await fetch("/api/identify", { method: "POST", body: form });
    if (!response.ok) throw new Error(await errorText(response));
    const result = await response.json();
    renderAnswer({ ...result, kind: result.plant ? "plant" : "unsure" });
    setStatus(result.plant ? "" : t("notSure"));
    if (result.plant) earnBadge(result.plant);
  } catch (error) {
    setStatus(`${error.message} ${t("stillAsk")}`);
  } finally {
    $("photo").value = "";
  }
}

async function loadPlants() {
  try {
    const response = await fetch("/api/plants");
    if (!response.ok) throw new Error(`Error ${response.status}`);
    plants = await response.json();
  } catch {
    setStatus(t("loadFail"));
  }
  renderSeason();
  renderAchievements();
}

function init() {
  $("ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    ask($("question").value);
  });
  $("photo").addEventListener("change", (event) => identify(event.target.files[0]));
  $("met").addEventListener("click", markFound);
  $("listen").addEventListener("click", toggleVoice);
  const toggle = () => setLanguage(isAs() ? "en" : "as");
  $("lang-toggle").addEventListener("click", toggle);
  $("card-lang").addEventListener("click", toggle);
  applyStaticText();
  renderMonth();
  renderJournal();
  loadPlants();
  loadHealth();
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
}

init();
