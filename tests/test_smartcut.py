import hashlib
import subprocess

import pytest

from moment_miner.export import export_clip
from moment_miner.ffbin import ffmpeg_exe, ffprobe_exe
from moment_miner.probe import keyframe_after, keyframe_before, probe

from .conftest import requires_ffmpeg

# fast_video is 10 s @ 50 fps with -g 25: keyframes on every 0.5 s, one
# frame is 0.02 s. The 15 fps fixture is too coarse to catch small errors —
# a four-frame mistake there is 0.27 s and hides inside any loose tolerance.
FRAME = 0.02


def _decoded(path: str, start: float, dur: float) -> bytes:
    """Raw decoded pixels — identity here is independent of container framing."""
    return subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{start}", "-i", path, "-t", f"{dur}",
         "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True,
    ).stdout


def _digest(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _stream_kinds(path: str) -> list[str]:
    out = subprocess.run(
        [ffprobe_exe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return out.split()


@requires_ffmpeg
def test_keyframe_helpers(fast_video):
    path = str(fast_video)
    assert abs(keyframe_before(path, 2.2) - 2.0) < 0.05
    assert abs(keyframe_after(path, 2.2) - 2.5) < 0.05
    assert keyframe_after(path, 9.99) is None


@requires_ffmpeg
def test_smart_cut_is_frame_accurate_at_both_ends(tmp_path, fast_video):
    out = tmp_path / "smart.mp4"
    _, a0, a1 = export_clip(str(fast_video), 2.2, 4.0, str(out), pad=0.0)

    assert abs(a0 - 2.2) < 1e-6 and abs(a1 - 4.0) < 1e-6
    # Both boundaries are cut, so the duration lands within a frame either way.
    assert abs(probe(str(out))["duration"] - 1.8) < 2 * FRAME


@requires_ffmpeg
def test_smart_cut_interior_is_the_untouched_source(tmp_path, fast_video):
    """The copied interior must decode identically AND sit at the right offset.

    Checking pixels alone passes even when the clip starts on the wrong
    frame; checking the offset is what catches a cut that is a few frames
    out. A hand-rolled ffmpeg version passed every other test here while
    starting four frames early.
    """
    out = tmp_path / "smart.mp4"
    export_clip(str(fast_video), 2.2, 4.0, str(out), pad=0.0)

    # 2.7 s is past the first keyframe after 2.2 s, so it is copied, not recoded.
    source = _decoded(str(fast_video), 2.7, 0.2)
    assert source, "fixture produced no frames"
    assert _digest(_decoded(str(out), 2.7 - 2.2, 0.2)) == _digest(source)


@requires_ffmpeg
def test_no_smart_snaps_back_to_a_keyframe(tmp_path, fast_video):
    out = tmp_path / "snapped.mp4"
    _, a0, _ = export_clip(str(fast_video), 2.2, 4.0, str(out), pad=0.0, smart=False)
    assert abs(a0 - 2.0) < 0.05
    assert probe(str(out))["duration"] > 1.8


@requires_ffmpeg
def test_smart_cut_video_only_source(tmp_path, silent_video):
    """Canon clips are not all the same shape: 2M4A2377.MP4 has no audio."""
    out = tmp_path / "silent_cut.mp4"   # not the fixture's own path
    _, a0, a1 = export_clip(str(silent_video), 1.0, 2.0, str(out), pad=0.0)
    assert abs(a0 - 1.0) < 1e-6 and abs(a1 - 2.0) < 1e-6
    assert _stream_kinds(str(out)) == ["video"]
    assert not probe(str(out))["has_audio"]


@requires_ffmpeg
def test_smart_cut_reports_failure(tmp_path):
    with pytest.raises(Exception):
        export_clip(str(tmp_path / "nope.mp4"), 1.0, 2.0, str(tmp_path / "o.mp4"))
