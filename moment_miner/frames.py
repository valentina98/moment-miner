import subprocess

import numpy as np

from .ffbin import ffmpeg_exe

# Keep the short side at or above the embedding processor's input: SigLIP2-256
# resizes to a flat 256x256 with no crop, so a 180-row raster is upsampled 1.42x
# and the vertical detail the encoder sees is invented. 16:9 is the source aspect,
# and rawvideo piping requires known dimensions.
FRAME_W, FRAME_H = 456, 256


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


def extract_stills(path: str, times, w: int, h: int) -> np.ndarray:
    """One RGB frame at each of `times` → (k, h, w, 3) uint8.

    A seek per still rather than a pass over the span: a caption wants two or
    three stills out of an 8 s window, and decoding the window at 4K is what
    this exists to avoid. A time past the last frame yields nothing, so k can
    be less than len(times).
    """
    size = w * h * 3
    out = []
    for t in times:
        proc = subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", path,
             "-vf", f"scale={w}:{h}", "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True,
        )
        if len(proc.stdout) >= size:
            out.append(np.frombuffer(proc.stdout[:size], np.uint8).reshape(h, w, 3))
    return np.stack(out) if out else np.empty((0, h, w, 3), np.uint8)


def _stream_frames(path: str, fps: float, w: int, h: int):
    """Yield (h, w, 3) frames from one sequential decode of the whole file."""
    proc = subprocess.Popen(
        [ffmpeg_exe(), "-v", "error", "-i", path,
         "-vf", f"fps={fps:.6f},scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def extract_windows(
    path: str, spans, n: int = 8, win: float = 8.0,
    w: int = FRAME_W, h: int = FRAME_H,
):
    """Yield (t0, t1, frames) per span, decoding the video once where it can.

    One ffmpeg per span makes every second of footage decode twice at 50%
    overlap, and each call pays its own process start and seek: 2074 s of the
    4846 s a 28-minute job spends, against 946 s for a single pass
    (`research/boundary-selection.md`). This cuts that without changing a
    single pixel.

    A span is served from the sequential pass only when the pass samples the
    same instants the per-span seek would: the span must be exactly `win` long
    (so `fps=n/span` equals the pass rate) and must start on a frame of the
    pass's own grid. Measured 2026-09-09 on `2M4A2407.MP4`: the three spans
    meeting that test came back byte-identical, and the tail span -- 11.64 s,
    on neither grid -- differed in all 8 frames at a mean absolute difference
    of 23.9 per channel. So a naive "just decode once" is *not* the free
    change it looks like; the tail is what makes it not free, and `windows()`
    gives every video exactly one such span. Across the 15-video corpus that
    is 15 of 140 spans (10.7%), and they keep their own seek here.

    `frames` is None for a span the file could not produce -- a truncated file
    ends the sequential pass early, the way it used to fail the seek from the
    first bad window on. Spans are yielded as the decode reaches them rather
    than in the order given; rows are keyed by segment id, so nothing
    downstream depends on the order.
    """
    spans = list(spans)
    rate = n / win
    direct, seq = [], {}
    for t0, t1 in spans:
        idx = t0 * rate
        if abs((t1 - t0) - win) < 1e-6 and abs(idx - round(idx)) < 1e-6:
            seq.setdefault(round(idx) + n - 1, []).append((t0, t1))
        else:
            direct.append((t0, t1))

    if seq:
        pending = dict(seq)
        buf: list[np.ndarray] = []
        for i, frame in enumerate(_stream_frames(path, rate, w, h)):
            buf.append(frame)
            if len(buf) > n:
                buf.pop(0)
            for t0, t1 in pending.pop(i, ()):
                yield t0, t1, np.stack(buf)
        for waiting in pending.values():
            for t0, t1 in waiting:
                yield t0, t1, None

    for t0, t1 in direct:
        try:
            yield t0, t1, extract_frames(path, t0, t1, n=n, w=w, h=h)
        except Exception:
            yield t0, t1, None

def frame_batches(path: str, fps: float, batch: int = 64,
                  w: int = FRAME_W, h: int = FRAME_H):
    """Yield (frames, times) from one sequential decode of the whole file.

    Batched because a dense pass over a long clip does not fit in memory: an hour at
    5 fps is 18k frames, which is 6 GB of raster at this size.
    """
    buf, ts = [], []
    for i, f in enumerate(_stream_frames(path, fps, w, h)):
        buf.append(f)
        ts.append(i / fps)
        if len(buf) == batch:
            yield np.stack(buf), np.asarray(ts, dtype=np.float64)
            buf, ts = [], []
    if buf:
        yield np.stack(buf), np.asarray(ts, dtype=np.float64)
