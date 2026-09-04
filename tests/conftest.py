import shutil
import subprocess

import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


@pytest.fixture
def synthetic_video(tmp_path):
    (tmp_path / "archive").mkdir()
    out = tmp_path / "archive" / "test_video.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=15",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "30",
         "-c:a", "aac", "-shortest", str(out)],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture
def fast_video(tmp_path):
    """50 fps, keyframe every 0.5 s — fine-grained enough to expose a cut
    that is only a few frames out. The 15 fps fixture is not: four frames
    there is 0.27 s, which hides inside any loose tolerance."""
    out = tmp_path / "fast.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=50",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "25",
         "-c:a", "aac", "-shortest", str(out)],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture
def silent_video(tmp_path):
    """No audio stream at all, like the Canon 2M4A2377.MP4 clip."""
    out = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=25",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "25", str(out)],
        check=True, capture_output=True,
    )
    return out
