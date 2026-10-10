// Run with: node --test tests/js
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");

const A = require("../../web/achievements.js");

const PLANTS = JSON.parse(fs.readFileSync(path.join(__dirname, "../../data/plants.json"), "utf8"));
const ALL_IDS = PLANTS.map((p) => p.id);

function scanAll(ids, date = "2026-10-07") {
  return ids.reduce((state, id) => A.recordScan(state, id, date), A.emptyState());
}

test("every plant belongs to exactly one collection", () => {
  const inSets = A.SETS.flatMap((set) => set.plants);
  assert.deepEqual([...inSets].sort(), [...ALL_IDS].sort());
  assert.equal(new Set(inSets).size, inSets.length);
});

test("recordScan returns a new state and keeps the first scan date", () => {
  const empty = A.emptyState();
  const once = A.recordScan(empty, "bogori", "2026-10-07");
  const twice = A.recordScan(once, "bogori", "2026-10-09");

  assert.deepEqual(empty.scans, {});
  assert.equal(once.scans.bogori, "2026-10-07");
  assert.equal(twice.scans.bogori, "2026-10-07");
});

test("the first scan earns First step", () => {
  assert.deepEqual(A.earnedBadges(scanAll(["bogori"]), ALL_IDS), ["first"]);
});

test("five scans earn the five-plant milestone", () => {
  const earned = A.earnedBadges(scanAll(ALL_IDS.slice(0, 5)), ALL_IDS);
  assert.ok(earned.includes("five"));
  assert.ok(!earned.includes("ten"));
});

test("finishing a collection earns its badge", () => {
  const sour = A.SETS.find((set) => set.id === "tenga");
  assert.ok(A.earnedBadges(scanAll(sour.plants), ALL_IDS).includes("tenga"));
  assert.ok(!A.earnedBadges(scanAll(sour.plants.slice(1)), ALL_IDS).includes("tenga"));
});

test("every plant earns Aita's Star", () => {
  const earned = A.earnedBadges(scanAll(ALL_IDS), ALL_IDS);
  assert.ok(earned.includes("star"));
  assert.ok(!A.earnedBadges(scanAll(ALL_IDS.slice(1)), ALL_IDS).includes("star"));
});

test("newBadges reports only what this scan unlocked", () => {
  const before = scanAll(ALL_IDS.slice(0, 4));
  const after = A.recordScan(before, ALL_IDS[4], "2026-10-07");

  assert.deepEqual(A.newBadges(before, after, ALL_IDS), { plant: ALL_IDS[4], awards: ["five"] });
  assert.deepEqual(A.newBadges(after, after, ALL_IDS), { plant: null, awards: [] });
});

test("progress counts scanned plants in a collection", () => {
  const greens = A.SETS.find((set) => set.id === "xaak");
  assert.deepEqual(A.progress(scanAll(greens.plants.slice(0, 2)), greens.plants), { done: 2, total: greens.plants.length });
});

test("parseState survives garbage from storage", () => {
  assert.deepEqual(A.parseState("not json"), A.emptyState());
  assert.deepEqual(A.parseState(JSON.stringify({ scans: ["x"] })), A.emptyState());
  assert.deepEqual(A.parseState(JSON.stringify({ scans: { bogori: "2026-10-07", bad: 5 } })).scans, { bogori: "2026-10-07" });
});
