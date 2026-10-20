import subprocess

import numpy as np
import pytest
from click.testing import CliRunner

from moment_miner.cli import main
from moment_miner.export import Encode, export_clip, shift_timecode
from moment_miner.ffbin import ffmpeg_exe
from moment_miner.probe import probe

from .conftest import requires_ffmpeg

W, H = 320, 240


def _frames(path) -> np.ndarray:
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", f"scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.int16)


def _closest(frame, source) -> int:
    return int(np.argmin(np.abs(source - frame).mean(axis=(1, 2))))


@pytest.fixture
def stamped_video(tmp_path):
    """50 fps with a start timecode and a creation time, as a camera writes them."""
    out = tmp_path / "stamped.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=6:size=320x240:rate=50", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=6", "-c:v", "libx264", "-preset", "ultrafast",
         "-g", "25", "-c:a", "aac", "-shortest", "-timecode", "10:00:00:00",
         "-metadata", "creation_time=2026-01-01T12:00:00Z", str(out)],
        check=True, capture_output=True)
    return out


@requires_ffmpeg
def test_a_re_encoded_clip_starts_and_ends_on_the_exact_frames(tmp_path, fast_video):
    out, a0, a1 = export_clip(str(fast_video), 2.2, 4.0, str(tmp_path), encode=Encode())
    source, clip = _frames(fast_video), _frames(out)
    assert (a0, a1) == (2.2, 4.0)
    assert len(clip) == 90
    assert _closest(clip[0], source) == 110 and _closest(clip[-1], source) == 199


@requires_ffmpeg
def test_every_setting_not_given_is_the_sources(tmp_path, stamped_video):
    out, _, _ = export_clip(str(stamped_video), 2.0, 3.0, str(tmp_path), encode=Encode())
    got, src = probe(out), probe(str(stamped_video))
    for key in ("codec", "pix_fmt", "width", "height", "fps", "has_audio", "audio_codec", "shot_at"):
        assert got[key] == src[key], key
    assert got["timecode"] == "10:00:02:00"


@requires_ffmpeg
def test_without_crf_the_clip_aims_at_the_sources_bitrate(tmp_path):
    # Noise, because on a clean test pattern x264 cannot spend the bitrate it
    # is given: a clip of plain testsrc lands at 0.4x the bitrate asked for.
    src = tmp_path / "noisy.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=duration=6:size=320x240:rate=50,noise=alls=20:allf=t",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "25", str(src)],
        check=True, capture_output=True)
    rates = {}
    for name, encode in [("source", Encode()), ("crf", Encode(crf=35))]:
        (tmp_path / name).mkdir()
        out, _, _ = export_clip(str(src), 1.0, 3.0, str(tmp_path / name), encode=encode)
        rates[name] = probe(out)["bit_rate"] / probe(str(src))["bit_rate"]
    assert 0.8 < rates["source"] < 1.5 and rates["crf"] < 0.5, rates


@requires_ffmpeg
def test_given_settings_replace_the_sources(tmp_path, stamped_video):
    out, _, _ = export_clip(str(stamped_video), 1.0, 3.0, str(tmp_path),
                            encode=Encode(size=(160, 120), codec="h265", fps=25.0, audio=False))
    got = probe(out)
    assert (got["codec"], got["width"], got["height"], got["fps"], got["has_audio"]) == (
        "hevc", 160, 120, 25.0, False)
    assert abs(got["duration"] - 2.0) < 0.05


@requires_ffmpeg
def test_prores_is_written_as_mov(tmp_path, fast_video):
    out, _, _ = export_clip(str(fast_video), 1.0, 1.5, str(tmp_path), encode=Encode(codec="prores"))
    assert out.endswith(".mov") and probe(out)["codec"] == "prores"


@pytest.mark.parametrize("tc, fps, seconds, want", [
    ("17:22:15:48", 50, 0.04, "17:22:16:00"),
    ("23:59:59:24", 25, 0.04, "00:00:00:00"),
    ("00:00:59;29", 29.97, 1 / 29.97, "00:01:00;02"),
    ("00:09:59;29", 29.97, 1 / 29.97, "00:10:00;00"),
    ("00:00:00;00", 29.97, 600, "00:10:00;00"),
    ("00:00:59;59", 59.94, 1 / 59.94, "00:01:00;04"),
])
def test_timecode_moves_on_by_the_cut_start(tc, fps, seconds, want):
    assert shift_timecode(tc, fps, seconds) == want


@pytest.mark.parametrize("args, message", [
    (["--crf", "18"], "need --re-encode"),
    (["--re-encode", "--no-smart"], "pick one"),
    (["--re-encode", "--codec", "prores", "--crf", "18"], "prores has no --crf"),
])
def test_settings_that_cannot_apply_stop_before_any_cut(tmp_path, args, message):
    project = tmp_path / "p.llc"
    project.write_text('{"version": 2, "mediaFileName": "v.mp4", "cutSegments": []}')
    result = CliRunner().invoke(main, ["--data-dir", str(tmp_path / "data"), "cut",
                                       str(project), *args])
    assert result.exit_code == 2 and message in result.output, result.output
