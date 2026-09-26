from moment_miner.calibrate import calibrate, estimate, load_rate
from moment_miner.embeddings.mock import MockBackend

from .conftest import requires_ffmpeg


@requires_ffmpeg
def test_calibration_is_stored_per_machine_and_backend(tmp_path):
    assert load_rate(tmp_path, "mock") is None
    entry = calibrate(tmp_path, MockBackend(), log=lambda *_: None,
                      clip={"size": "320x180", "rate": 15, "seconds": 12})
    assert entry["segments"] == 2 and entry["s_per_segment"] > 0
    assert load_rate(tmp_path, "mock") == entry
    assert load_rate(tmp_path, "siglip") is None


def test_estimate_counts_segments_the_indexer_would_cut(monkeypatch):
    import moment_miner.calibrate as cal
    monkeypatch.setattr(cal, "probe", lambda path: {"duration": {"a": 24.0, "b": 4.0}[path]})
    segments, seconds, eta = estimate(
        [{"path": "a"}, {"path": "b"}], {"s_per_segment": 10.0}, win=8.0, stride=4.0)
    assert (segments, seconds, eta) == (5 + 1, 28.0, 60.0)
