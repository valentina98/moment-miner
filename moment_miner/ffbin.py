import functools
import shutil
import subprocess
from pathlib import Path

# How long one ffmpeg/ffprobe call may take, or a streaming decode may go
# without producing a frame. Every call here does bounded work -- a probe, a
# seek, one window, one clip -- so a call this slow is a dead drive or a file
# the decoder loops on, not a slow machine.
TIMEOUT_S = 300.0


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`")


@functools.lru_cache(maxsize=1)
def ffprobe_exe() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path
    raise RuntimeError("ffprobe not found: install ffmpeg system-wide")


class FFmpegTimeout(RuntimeError):
    pass


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run with TIMEOUT_S; the hung process is killed, not left behind."""
    try:
        return subprocess.run(cmd, timeout=TIMEOUT_S, **kwargs)
    except subprocess.TimeoutExpired:
        raise FFmpegTimeout(
            f"{Path(cmd[0]).name} gave no result in {TIMEOUT_S:.0f} s") from None
