import shutil
import time

import pytest

from moment_miner import ffbin, frames, probe

from .conftest import requires_ffmpeg


def _fake(tmp_path, name, real):
    """A stand-in binary that never returns for a file named hang.mp4."""
    exe = tmp_path / name
    exe.write_text(
        "#!/bin/sh\n"
        'case "$*" in */hang.mp4*) exec sleep 60;; esac\n'
        f'exec {real} "$@"\n')
    exe.chmod(0o755)
    return str(exe)


def test_run_kills_a_call_that_never_returns(monkeypatch):
    monkeypatch.setattr(ffbin, "TIMEOUT_S", 0.5)
    t = time.monotonic()
    with pytest.raises(ffbin.FFmpegTimeout, match="sleep gave no result"):
        ffbin.run(["sleep", "60"])
    assert time.monotonic() - t < 5


def test_a_stalled_stream_raises_instead_of_ending_quietly(tmp_path, monkeypatch):
    monkeypatch.setattr(ffbin, "TIMEOUT_S", 0.5)
    fake = _fake(tmp_path, "ffmpeg", "false")
    monkeypatch.setattr(frames, "ffmpeg_exe", lambda: fake)
    with pytest.raises(ffbin.FFmpegTimeout, match="no frame"):
        list(frames._stream_frames(str(tmp_path / "hang.mp4"), 1.0, 16, 16))


@requires_ffmpeg
def test_a_slow_consumer_is_not_a_stall(synthetic_video, monkeypatch):
    monkeypatch.setattr(ffbin, "TIMEOUT_S", 0.5)
    got = 0
    for _ in frames._stream_frames(str(synthetic_video), 1.0, 32, 32):
        got += 1
        if got <= 2:
            time.sleep(1.0)
    assert got >= 9


@requires_ffmpeg
def test_a_hanging_file_is_marked_failed_and_the_run_completes(
        tmp_path, synthetic_video, monkeypatch):
    from moment_miner.embeddings.mock import MockBackend
    from moment_miner.indexer import index_pending
    from moment_miner.manifest import Manifest
    from moment_miner.store import SegmentStore

    shutil.copy(synthetic_video, synthetic_video.parent / "hang.mp4")
    monkeypatch.setattr(ffbin, "TIMEOUT_S", 1.0)
    fake_probe = _fake(tmp_path, "ffprobe", shutil.which("ffprobe"))
    fake_mpeg = _fake(tmp_path, "ffmpeg", shutil.which("ffmpeg"))
    monkeypatch.setattr(probe, "ffprobe_exe", lambda: fake_probe)
    monkeypatch.setattr(frames, "ffmpeg_exe", lambda: fake_mpeg)
    manifest = Manifest(tmp_path / "data" / "manifest.db")
    manifest.scan(synthetic_video.parent)

    t = time.monotonic()
    counts = index_pending(manifest, SegmentStore(tmp_path / "data", "mock"),
                           MockBackend(), use_asr=False, log=lambda *a: None)

    # The fake sleeps 60 s; without the timeout the run waits it out.
    assert time.monotonic() - t < 30

    assert counts["indexed"] == 1 and counts["errors"] == 1
    [failed] = manifest.errors()
    assert failed["path"].endswith("hang.mp4")
    assert "FFmpegTimeout" in failed["error"]


def _hangs(*a, **k):
    raise ffbin.FFmpegTimeout("ffmpeg gave no result in 300 s")


def test_a_timeout_on_an_off_grid_span_fails_the_file(monkeypatch):
    monkeypatch.setattr(frames, "extract_frames", _hangs)
    # Shorter than the window, so the span is decoded on its own, not streamed.
    with pytest.raises(ffbin.FFmpegTimeout):
        list(frames.extract_windows("tail.mp4", [(0.0, 3.0)], n=4, win=8.0))


def test_a_calibration_timeout_drops_the_estimate_not_the_run(monkeypatch, capsys):
    from moment_miner import calibrate, cli

    monkeypatch.setattr(calibrate, "load_rate", lambda *a: None)
    monkeypatch.setattr(calibrate, "calibrate", _hangs)
    backend = type("Backend", (), {"name": "mock"})()
    assert cli._estimate("data", backend, [{"path": "hang.mp4"}], 4.0, 2.0, False) is None
    assert "estimate: unavailable" in capsys.readouterr().out
