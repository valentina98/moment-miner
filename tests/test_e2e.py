import subprocess

import numpy as np
import pytest

from moment_miner.embeddings.mock import MockBackend
from moment_miner.frames import extract_frames, extract_windows
from moment_miner.export import export_clip
from moment_miner.indexer import index_pending, windows
from moment_miner.manifest import Manifest
from moment_miner.probe import probe
from moment_miner.store import SegmentStore

from .conftest import requires_ffmpeg


def test_windows_step_forward_and_the_last_one_ends_at_the_video_end():
    """Starts are multiples of the stride; whatever is left over is covered by
    a final full-length segment ending exactly at the video's end."""
    assert list(windows(10.0, win=8.0, stride=4.0)) == [(0.0, 8.0), (2.0, 10.0)]
    assert list(windows(19.64, win=8.0, stride=4.0))[-1] == (11.64, 19.64)


def test_no_segment_is_a_fragment():
    """An unguarded forward walk emitted a segment wherever any video remained,
    so a 4.04 s clip produced a 40 ms second segment -- a full index row, a
    duplicate of the tail before it, and something a caption pass pays for."""
    for duration in (4.04, 8.5, 9.92, 11.92, 19.64, 47.0):
        spans = [t1 - t0 for t0, t1 in windows(duration, win=8.0, stride=4.0)]
        assert min(spans) >= min(8.0, duration) - 1e-9, duration


def test_a_clip_shorter_than_one_window_is_a_single_segment():
    assert list(windows(4.04, win=8.0, stride=4.0)) == [(0.0, 4.04)]


def test_a_small_tail_stretches_the_last_segment_instead_of_adding_one():
    """Adding a segment for a sub-stride tail duplicates the one before it;
    that made 4.9% of all segments redundant across footage/."""
    assert list(windows(8.5, win=8.0, stride=4.0)) == [(0.0, 8.5)]
    assert list(windows(9.92, win=8.0, stride=4.0)) == [(0.0, 9.92)]
    # A tail worth its own segment still gets one.
    assert list(windows(10.0, win=8.0, stride=4.0)) == [(0.0, 8.0), (2.0, 10.0)]


def test_the_stretched_segment_keeps_the_video_end():
    """The stretched segment keeps the video's end rather than start + win, so
    the final fraction of a second stays inside a segment."""
    assert list(windows(8.072, win=8.0, stride=4.0))[-1][1] == pytest.approx(8.072)


def test_every_second_of_video_is_covered():
    for duration in (4.04, 8.5, 10.0, 19.64, 47.0):
        w = list(windows(duration, win=8.0, stride=4.0))
        assert w[0][0] == 0.0 and w[-1][1] == pytest.approx(duration)
        for (_, prev_end), (next_start, _) in zip(w, w[1:]):
            assert next_start <= prev_end, duration


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

    real = idx.extract_windows

    def fails_past_8s(path, spans, n, win):
        for t0, t1, frames in real(path, spans, n=n, win=win):
            yield t0, t1, (None if t0 >= 8.0 else frames)

    monkeypatch.setattr(idx, "extract_windows", fails_past_8s)

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

    def always_fails(path, spans, n, win):
        return ((t0, t1, None) for t0, t1 in spans)

    monkeypatch.setattr(idx, "extract_windows", always_fails)

    be = MockBackend()
    manifest = Manifest(tmp_path / "mm.db")
    store = SegmentStore(tmp_path / "data", be.name)
    manifest.scan(synthetic_video.parent)

    result = index_pending(manifest, store, be, use_asr=False, log=lambda *_: None)
    assert result["indexed"] == 0 and result["errors"] == 1 and result["segments"] == 0


@requires_ffmpeg
def test_one_decode_pass_returns_the_same_pixels_as_per_span_seeks(tmp_path):
    """The whole point of decoding once is that nothing else changes.

    A span served from the sequential pass and the same span seeked to
    directly must agree byte for byte, or the speedup is a silent re-index
    and a silent eval movement. 11.5 s is chosen to exercise both routes:
    (0, 8) sits on the pass's own frame grid, (3.5, 11.5) does not and keeps
    its seek. Measured 2026-09-09 on real footage, an off-grid span differed
    in all 8 frames at a mean absolute difference of 23.9 per channel, so
    this is a real failure mode and not a theoretical one.
    """
    out = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=11.5:size=320x240:rate=15",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "30", str(out)],
        check=True, capture_output=True,
    )
    spans = list(windows(11.5, win=8.0, stride=4.0))
    assert spans == [(0.0, 8.0), (3.5, 11.5)], "fixture no longer covers both routes"

    got = {(t0, t1): f for t0, t1, f in extract_windows(str(out), spans, n=8, win=8.0)}
    assert set(got) == set(spans)
    for t0, t1 in spans:
        want = extract_frames(str(out), t0, t1, n=8)
        assert got[(t0, t1)] is not None
        assert np.array_equal(got[(t0, t1)], want), f"span {t0}-{t1} changed"


@requires_ffmpeg
def test_one_decode_pass_spawns_one_ffmpeg_for_the_aligned_spans(tmp_path, monkeypatch):
    """The saving is process count: 420 seeks become one pass plus the tail.
    Counting them is the only way this stays true under later edits."""
    import moment_miner.frames as fr

    out = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=23.5:size=320x240:rate=15",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "30", str(out)],
        check=True, capture_output=True,
    )
    spans = list(windows(23.5, win=8.0, stride=4.0))
    assert len(spans) == 5

    seeks = 0
    real = fr.extract_frames

    def counted(*a, **k):
        nonlocal seeks
        seeks += 1
        return real(*a, **k)

    monkeypatch.setattr(fr, "extract_frames", counted)
    list(fr.extract_windows(str(out), spans, n=8, win=8.0))
    assert seeks == 1, f"{len(spans)} spans should cost 1 seek, not {seeks}"
