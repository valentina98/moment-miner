import subprocess

import numpy as np

from .ffbin import ffmpeg_exe

# Small fixed raster: embedding processors resize to square anyway, and
# rawvideo piping requires known dimensions.
FRAME_W, FRAME_H = 320, 180


def extract_frames(
    path: str, t0: float, t1: float, n: int = 8,
    w: int = FRAME_W, h: int = FRAME_H,
) -> np.ndarray:
    """Uniformly sample up to n RGB frames from [t0, t1] → (k, h, w, 3) uint8."""
    dur = max(t1 - t0, 0.1)
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
         "-i", path, "-vf", f"fps={n / dur:.6f},scale={w}:{h}",
         "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )
    buf = np.frombuffer(proc.stdout, np.uint8)
    k = len(buf) // (w * h * 3)
    return buf[: k * w * h * 3].reshape(k, h, w, 3)
