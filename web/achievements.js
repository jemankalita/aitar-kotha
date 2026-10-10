"use strict";

// Badge rules for Aitar Kotha. Pure functions over a small state object, so they are easy to test
// (node --test tests/js) and the page only has to store and render the result.
// State: { version: 1, scans: { plantId: "YYYY-MM-DD" } } — one entry per plant scanned by photo.

(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.AitaAchievements = api;
  }
})(typeof self !== "undefined" ? self : this, function () {
  const SETS = [
    {
      id: "tenga",
      as: "টেঙাৰ দল",
      en: "The sour ones",
      plants: ["ou_tenga", "thekera", "kordoi", "tenga_mora", "kaji_nemu", "amlokhi", "robab_tenga"],
    },
    {
      id: "bihu",
      as: "বিহুৰ গছ",
      en: "Bihu plants",
      plants: ["kopou_phul", "lau", "dighloti", "haladhi", "tamul", "paan", "dhan", "tulsi", "bah"],
    },
    {
      id: "xaak",
      as: "পাকঘৰৰ শাক",
      en: "Kitchen greens",
      plants: ["manimuni", "masundari", "dhekia", "kolmou", "bhedai_lota", "kosu", "xoriyoh"],
    },
    {
      id: "fol",
      as: "চোতালৰ ফল",
      en: "Courtyard fruits",
      plants: ["bogori", "leteku", "poniol", "jolphai", "jamu", "modhuri_am", "bel", "kol"],
    },
    {
      id: "habi",
      as: "হাবি আৰু শিল্প",
      en: "Forest and craft",
      plants: ["hollong", "simolu", "nahor", "som", "era", "sasi", "tokou", "chah", "nim", "padum", "bhut_jolokia", "sewali"],
    },
  ];

  const MILESTONES = [
    { id: "first", as: "প্ৰথম খোজ", en: "First step", need: 1 },
    { id: "five", as: "পাঁচবিধ", en: "Five plants", need: 5 },
    { id: "ten", as: "দহবিধ", en: "Ten plants", need: 10 },
    { id: "twenty", as: "বিশবিধ", en: "Twenty plants", need: 20 },
  ];

  const STAR = { id: "star", as: "আইতাৰ তৰা", en: "Aita's Star" };
  const DATE = /^\d{4}-\d{2}-\d{2}$/;

  function emptyState() {
    return { version: 1, scans: {} };
  }

  function recordScan(state, plantId, date) {
    if (state.scans[plantId]) return state;
    return { ...state, scans: { ...state.scans, [plantId]: date } };
  }

  function scannedCount(state) {
    return Object.keys(state.scans).length;
  }

  function progress(state, plantIds) {
    return { done: plantIds.filter((id) => state.scans[id]).length, total: plantIds.length };
  }

  function earnedBadges(state, allPlantIds) {
    const count = scannedCount(state);
    const milestones = MILESTONES.filter((m) => count >= m.need).map((m) => m.id);
    const sets = SETS.filter((set) => set.plants.every((id) => state.scans[id])).map((set) => set.id);
    const star = allPlantIds.length > 0 && allPlantIds.every((id) => state.scans[id]) ? [STAR.id] : [];
    return [...milestones, ...sets, ...star];
  }

  function newBadges(before, after, allPlantIds) {
    const plant = Object.keys(after.scans).find((id) => !before.scans[id]) || null;
    const had = new Set(earnedBadges(before, allPlantIds));
    const awards = earnedBadges(after, allPlantIds).filter((id) => !had.has(id));
    return { plant, awards };
  }

  function parseState(text) {
    try {
      const raw = JSON.parse(text);
      if (!raw || typeof raw.scans !== "object" || Array.isArray(raw.scans)) return emptyState();
      const scans = Object.fromEntries(
        Object.entries(raw.scans).filter(([, date]) => typeof date === "string" && DATE.test(date)),
      );
      return { version: 1, scans };
    } catch {
      return emptyState();
    }
  }

  function award(id) {
    return MILESTONES.find((m) => m.id === id) || SETS.find((s) => s.id === id) || (id === STAR.id ? STAR : null);
  }

  return { SETS, MILESTONES, STAR, emptyState, recordScan, scannedCount, progress, earnedBadges, newBadges, parseState, award };
});
