/* ===========================================================================
 * DataMill Analytics - Service Worker
 * Servido en la RAIZ (/sw.js) para tener scope '/' sobre toda la app.
 * Estrategia:
 *   - install : precachea la "app shell" (home + assets clave).
 *   - activate: limpia caches de versiones antiguas.
 *   - fetch   : network-first para navegaciones (Dash es dinamico) con fallback
 *               offline a la shell cacheada; cache-first + refresco en segundo
 *               plano para assets estaticos. NUNCA cachea los callbacks de Dash.
 * Sube CACHE_VERSION para invalidar la cache al desplegar cambios.
 * =========================================================================== */
const CACHE_VERSION = "datamill-v2";
const CORE_ASSETS = [
  "/",
  "/assets/custom.css",
  "/assets/manifest.json",
  "/assets/icons/icon-192.png",
  "/assets/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then((cache) => cache.addAll(CORE_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;                 // ignora POST (callbacks Dash)

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;  // solo mismo origen
  // No interferir con los endpoints dinamicos de Dash.
  if (url.pathname.startsWith("/_dash-update-component")) return;

  // Navegaciones (documento HTML): network-first, fallback a la shell offline.
  if (req.mode === "navigate") {
    event.respondWith(fetch(req).catch(() => caches.match("/")));
    return;
  }

  // Assets estaticos: cache-first y, en paralelo, refresco en segundo plano.
  event.respondWith(
    caches.match(req).then((cached) => {
      const network = fetch(req)
        .then((resp) => {
          if (resp && resp.status === 200 && resp.type === "basic") {
            const copy = resp.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(req, copy));
          }
          return resp;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});
