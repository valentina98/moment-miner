import json
import threading
import urllib.request

import pytest

from moment_miner.embeddings.mock import MockBackend
from moment_miner.serve import make_server
from moment_miner.store import SegmentStore


@pytest.fixture
def server(tmp_path):
    be = MockBackend()
    store = SegmentStore(tmp_path, be.name)
    store.add([{
        "id": "v1:0", "video_id": "v1", "path": "/x/v1.mp4",
        "t0": 4.0, "t1": 12.0, "text": "kong vault",
        "vector": be.embed_text(["kong vault"])[0],
    }])
    store.rebuild_fts()
    srv = make_server(be, store, host="127.0.0.1", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, json.loads(r.read())


def test_health(server):
    status, body = _get(f"{server}/health")
    assert status == 200
    assert body["status"] == "ok" and body["backend"] == "mock"
    assert body["segments"] == 1


def test_search_endpoint(server):
    status, body = _get(f"{server}/search?q=kong%20vault&k=5")
    assert status == 200
    assert body["results"][0]["path"] == "/x/v1.mp4"
    assert body["results"][0]["t0"] == 4.0


def test_search_requires_query(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{server}/search")
    assert e.value.code == 400
