import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .embeddings.base import EmbeddingBackend
from .search import search as run_search
from .store import SegmentStore

RESULT_FIELDS = ("path", "t0", "t1", "score", "text")


def make_server(
    backend: EmbeddingBackend,
    store: SegmentStore,
    host: str = "127.0.0.1",
    port: int = 7700,
) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                self._json(200, {
                    "status": "ok",
                    "backend": backend.name,
                    "device": getattr(backend, "device", "cpu"),
                    "segments": store.count(),
                })
            elif url.path == "/search":
                params = parse_qs(url.query)
                query = params.get("q", [""])[0].strip()
                if not query:
                    self._json(400, {"error": "missing query parameter q"})
                    return
                k = int(params.get("k", ["10"])[0])
                with lock:
                    hits = run_search(query, backend, store, k=k)
                self._json(200, {
                    "query": query,
                    "results": [{f: h.get(f) for f in RESULT_FIELDS} for h in hits],
                })
            else:
                self._json(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)
