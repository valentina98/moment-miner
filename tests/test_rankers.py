from moment_miner.rankers import get_ranker, ranker_names


def test_the_shipped_rankers_are_listed():
    assert {"mock", "laya", "jev"} <= set(ranker_names())


def test_mock_scores_by_word_overlap():
    r = get_ranker("mock")
    assert r.score("pull ups", ["pull ups on bar", "vault over rail"]) == [1.0, 0.0]


def test_top_reads_the_axis_it_is_given():
    r = get_ranker("mock")
    rows = [{"action": "vault over rail", "scene": "a park"},
            {"action": "pull up", "scene": "vault street"}]
    assert r.top("vault", rows, k=1, axis="action")[0]["action"] == "vault over rail"
    assert r.top("vault", rows, k=1, axis="scene")[0]["scene"] == "vault street"


def test_jev_refuses_without_a_key():
    import pytest
    r = get_ranker("jev")
    r._key = None
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        r.score("x", ["y"])


def test_jev_names_the_model_it_asks(monkeypatch):
    import io
    import json
    import urllib.request
    sent = {}

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data))
        return io.BytesIO(b'{"answers": {"q0": {"noul": 0.8}}}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    r = get_ranker("jev")
    r._key = "test"
    assert r.score("a vault", ["vault over wall"]) == [0.8]
    assert sent["model"] == "jev-latest"
