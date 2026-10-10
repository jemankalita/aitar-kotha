"use strict";

// Keeps the app shell and the family notes on the phone, so asking by name works with no signal.
const CACHE = "aita-v5";
const SHELL = [
  "./",
  "index.html",
  "app.js",
  "i18n.js",
  "achievements.js",
  "styles.css",
  "manifest.webmanifest",
  "icon.svg",
  "fonts/fraunces-latin.woff2",
  "fonts/noto-serif-bengali.woff2",
];
const NOTES = "/api/plants";

async function install() {
  const cache = await caches.open(CACHE);
  await cache.addAll(SHELL);
  try {
    await cache.add(NOTES); // a flaky first load must not cost us the app shell
  } catch {
    // The notes get cached on the next successful visit (network-first below).
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(install());
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))),
  );
  self.clients.claim();
});

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = await cache.match(request, { ignoreSearch: request.mode === "navigate" });
    if (cached) return cached;
    if (request.mode === "navigate") {
      const shell = await cache.match("index.html");
      if (shell) return shell;
    }
    throw error;
  }
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return; // questions and photos always go to the server
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  event.respondWith(networkFirst(request));
});
