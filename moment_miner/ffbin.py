import functools
import os
import shutil
import subprocess
import time
from pathlib import Path

# How long one ffmpeg/ffprobe call may take, or a streaming decode or a stream
# copy may go without producing anything. Every timed call does bounded work --
# a probe, a seek, one window -- so a call this slow is a dead drive or a file
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


def run_while_growing(cmd: list[str], out: str | Path, **kwargs) -> subprocess.CompletedProcess:
    """Like `run`, for a call whose length grows with its input -- a stream copy
    of a 15 GB file outlasts any fixed limit. It is killed only once `out` stops
    growing for TIMEOUT_S."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    size, since = None, time.monotonic()
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=min(1.0, TIMEOUT_S / 2))
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            now = os.path.getsize(out) if os.path.exists(out) else None
            if now != size:
                size, since = now, time.monotonic()
            elif time.monotonic() - since > TIMEOUT_S:
                proc.kill()
                proc.communicate()
                raise FFmpegTimeout(
                    f"{Path(cmd[0]).name} wrote nothing in {TIMEOUT_S:.0f} s") from None
