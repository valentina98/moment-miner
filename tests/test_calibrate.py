from moment_miner.calibrate import calibrate, estimate, load_rate
from moment_miner.embeddings.mock import MockBackend

from .conftest import requires_ffmpeg


@requires_ffmpeg
def test_calibration_is_stored_per_machine_and_backend(tmp_path, synthetic_video):
    """The sample is the first 8 s of a 10 s video: one segment."""
    data = tmp_path / "data"
    assert load_rate(data, "mock") is None
    entry = calibrate(data, MockBackend(), synthetic_video, log=lambda *_: None, seconds=8)
    assert entry["segments"] == 1 and entry["s_per_segment"] > 0
    assert load_rate(data, "mock") == entry
    assert load_rate(data, "siglip") is None


def test_estimate_counts_segments_the_indexer_would_cut(monkeypatch):
    import moment_miner.calibrate as cal
    monkeypatch.setattr(cal, "probe", lambda path: {"duration": {"a": 24.0, "b": 4.0}[path]})
    segments, seconds, eta = estimate(
        [{"path": "a"}, {"path": "b"}], {"s_per_segment": 10.0}, win=8.0, stride=4.0)
    assert (segments, seconds, eta) == (5 + 1, 28.0, 60.0)
