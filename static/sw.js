const CACHE_NAME = 'charles-v1';

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  // Pass-through padrão para permitir chamadas dinâmicas à API
  event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
});
