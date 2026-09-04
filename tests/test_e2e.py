from moment_miner.embeddings.mock import MockBackend
from moment_miner.export import export_clip
from moment_miner.indexer import index_pending, windows
from moment_miner.manifest import Manifest
from moment_miner.probe import probe
from moment_miner.store import SegmentStore

from .conftest import requires_ffmpeg


def test_windows():
    w = list(windows(10.0, win=8.0, stride=4.0))
    assert w == [(0.0, 8.0), (4.0, 10.0), (8.0, 10.0)]


@requires_ffmpeg
def test_index_and_export(tmp_path, synthetic_video):
    be = MockBackend()
    manifest = Manifest(tmp_path / "mm.db")
    store = SegmentStore(tmp_path / "data", be.name)
    manifest.scan(synthetic_video.parent)

    result = index_pending(manifest, store, be, use_asr=False, log=lambda *_: None)
    assert result == {"indexed": 1, "errors": 0, "segments": result["segments"]}
    assert result["segments"] >= 2
    assert store.count() == result["segments"]

    info = probe(str(synthetic_video))
    assert 9.5 < info["duration"] < 10.5

    hits = store.vector_search(be.embed_text(["anything"])[0], k=3)
    assert all(0 <= h["t0"] < h["t1"] <= 10.5 for h in hits)

    out = tmp_path / "clip.mp4"
    _, a0, a1 = export_clip(str(synthetic_video), 3.0, 6.0, str(out),
                            pad=0.5, snap=True, smart=False)
    assert out.stat().st_size > 1024
    # GOP is 30 frames @ 15fps → keyframes at 0/2/4s; start 2.5s must snap to 2.0
    assert abs(a0 - 2.0) < 0.1 and a1 >= 6.5
    assert abs(probe(str(out))["duration"] - (a1 - a0)) < 1.0

    # Re-scan: nothing new, nothing pending.
    assert manifest.scan(synthetic_video.parent)["unchanged"] == 1
    assert manifest.pending() == []


@requires_ffmpeg
def test_truncated_video_keeps_its_readable_prefix(tmp_path, synthetic_video, monkeypatch):
    """A truncated file fails only from the window past its last packet on.
    Discarding the whole video there loses every readable window before it —
    which is what a real 200 s clip, truncated at 164 s, cost us."""
    import moment_miner.indexer as idx

    real = idx.extract_frames

    def fails_past_8s(path, t0, t1, n):
        if t0 >= 8.0:
            raise RuntimeError("simulated truncation")
        return real(path, t0, t1, n=n)

    monkeypatch.setattr(idx, "extract_frames", fails_past_8s)

    be = MockBackend()
    manifest = Manifest(tmp_path / "mm.db")
    store = SegmentStore(tmp_path / "data", be.name)
    manifest.scan(synthetic_video.parent)

    result = index_pending(manifest, store, be, use_asr=False, log=lambda *_: None)
    assert result["indexed"] == 1 and result["errors"] == 0
    assert result["segments"] == 2
    assert all(h["t0"] < 8.0 for h in store.vector_search(be.embed_text(["x"])[0], k=10))


@requires_ffmpeg
def test_wholly_unreadable_video_is_an_error(tmp_path, synthetic_video, monkeypatch):
    """Skipping bad windows must not turn an unreadable file into a silent
    success with zero segments."""
    import moment_miner.indexer as idx

    def always_fails(path, t0, t1, n):
        raise RuntimeError("simulated corruption")

    monkeypatch.setattr(idx, "extract_frames", always_fails)

    be = MockBackend()
    manifest = Manifest(tmp_path / "mm.db")
    store = SegmentStore(tmp_path / "data", be.name)
    manifest.scan(synthetic_video.parent)

    result = index_pending(manifest, store, be, use_asr=False, log=lambda *_: None)
    assert result["indexed"] == 0 and result["errors"] == 1 and result["segments"] == 0
