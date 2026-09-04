import functools
import shutil


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
