import os
import time
from collections import defaultdict
from pathlib import Path

from .captions import (
    DEFAULT_FRAMES,
    DEFAULT_SHORT_FRAMES,
    LONG_VIDEO_S,
    SHORT_VIDEO_S,
    CaptionBackend,
    CaptionSidecar,
    frames_for_duration,
    still_times,
)
from .embeddings.base import EmbeddingBackend
from .frames import FRAME_H, FRAME_W, extract_stills, extract_windows
from .manifest import Manifest
from .motion import moving_fraction
from .probe import probe
from .store import MotionStore, SegmentStore

DEFAULT_WIN = 8.0
DEFAULT_STRIDE = 4.0
DEFAULT_FRAMES_PER_WINDOW = 8


def index_settings(
    backend_name: str,
    win: float = DEFAULT_WIN,
    stride: float = DEFAULT_STRIDE,
    frames_per_window: int = DEFAULT_FRAMES_PER_WINDOW,
) -> dict:
    """Everything an index depends on, as `Manifest` records it per video.

    The raster is read from `frames` rather than taken as an argument because
    nothing passes one: it is a constant of the embedding processor's input
    size, and recording it is how a change to that constant reaches an index
    already on disk.
    """
    return {
        "win": win,
        "stride": stride,
        "frames_per_window": frames_per_window,
        "frame_w": FRAME_W,
        "frame_h": FRAME_H,
        "backend": backend_name,
    }


def windows(duration: float, win: float = 8.0, stride: float = 4.0):
    """Cut a video into overlapping segments, stepping forward from 0.

    Windows start at 0 and step forward one stride at a time for as long as a
    whole window fits. Segment starts are therefore multiples of the stride,
    and which segment holds a timestamp can be read off without knowing the
    clip's duration.

    The tail needs its own case, the way the caption sampler needs one for
    spans under a stride. An unguarded forward walk emitted a segment wherever
    any video remained, so a 4.04 s clip produced a second segment covering
    4.00-4.04: forty milliseconds, indexed and searchable as a full row, and
    paid for by a caption pass. Measured 2026-09-06, a third of the local index
    and 40% of footage/cuts/ were such fragments, each a duplicate of the tail
    before it.

    So a leftover under half a stride stretches the last segment forward to the
    video's end, and a larger leftover gets its own full-length segment ending
    there. Either way every segment is full length and the video's final frame
    is inside one.

    Rewritten 2026-09-08 from an end-anchored backward walk, which produced the
    same spans for every duration the tests pin but read backwards and put the
    special case at the head. Valya's call: the forward walk is what a reader
    expects.
    """
    if duration <= win:
        yield 0.0, duration
        return
    spans, t = [], 0.0
    while t + win <= duration:
        spans.append((round(t, 6), round(t + win, 6)))
        t += stride
    tail = duration - spans[-1][1]
    if tail > 0:
        if tail < stride / 2:
            # Stretching costs a longer segment; adding one costs a near
            # duplicate -- that made 4.9% of all segments redundant.
            spans[-1] = (spans[-1][0], duration)
        else:
            spans.append((round(duration - win, 6), duration))
    yield from sorted(set(spans))


def index_pending(
    manifest: Manifest,
    store: SegmentStore,
    backend: EmbeddingBackend,
    use_asr: bool = True,
    asr_model: str = "small",
    win: float = DEFAULT_WIN,
    stride: float = DEFAULT_STRIDE,
    frames_per_window: int = DEFAULT_FRAMES_PER_WINDOW,
    caption_backend: CaptionBackend | None = None,
    caption_dir=None,
    motion_store: MotionStore | None = None,
    caption_frames: int | None = None,
    caption_frames_short: int | None = None,
    caption_frames_long: int | None = None,
    short_video_s: float = SHORT_VIDEO_S,
    long_video_s: float = LONG_VIDEO_S,
    log=print,
) -> dict:
    counts = {"indexed": 0, "errors": 0, "segments": 0}
    if caption_backend is not None:
        counts["captioned"] = 0
    if caption_backend is not None:
        caption_backend.preflight()
    settings = index_settings(backend.name, win, stride, frames_per_window)
    videos = manifest.pending(settings)
    for i, row in enumerate(videos, 1):
        vid, path = row["id"], row["path"]
        t_start = time.time()
        try:
            info = probe(path)
            duration = info["duration"]
            log(f"[{i}/{len(videos)}] {path} ({duration:.0f}s)")
            if use_asr and info["has_audio"]:
                from .asr import transcribe

                manifest.add_transcript(vid, transcribe(path, model_size=asr_model))
            sidecar = None
            n_caption_frames = frames_for_duration(
                duration, short=caption_frames_short, normal=caption_frames,
                long=caption_frames_long, short_video_s=short_video_s,
                long_video_s=long_video_s)
            if caption_backend is not None:
                sidecar = CaptionSidecar(path, caption_dir)
                sidecar.check_writable()
            rows, motion_rows = [], []
            unreadable = 0
            spans = windows(duration, win, stride)
            for t0, t1, frames in extract_windows(
                path, spans, n=frames_per_window, win=win
            ):
                if frames is None:
                    # A truncated file stops yielding from its last good packet
                    # on; everything before it is still worth indexing.
                    unreadable += 1
                    continue
                if len(frames) == 0:
                    continue
                seg_id = f"{vid}:{t0:.1f}"
                text = manifest.transcript_between(vid, t0, t1)
                if sidecar is not None:
                    caption = sidecar.get(seg_id)
                    if caption is None:
                        caption = caption_backend.caption(
                            frames, t1 - t0, n_caption_frames, stride)
                        sidecar.put(seg_id, t0, t1, caption, caption_backend.name)
                        counts["captioned"] += 1
                    text = f"{text} {caption}".strip()
                rows.append({
                    "id": seg_id,
                    "video_id": vid,
                    "path": path,
                    "t0": t0,
                    "t1": t1,
                    "text": text,
                    "vector": backend.embed_segment(frames),
                })
                # Measured here because the frames are already decoded;
                # extraction is ~20% of index time and a second pass would
                # pay it again. Only the raw value is kept -- see MotionStore.
                motion_rows.append({
                    "id": seg_id, "path": path, "t0": t0, "t1": t1,
                    "motion": moving_fraction(frames),
                })
            if not rows:
                raise RuntimeError(f"no readable windows ({unreadable} failed)")
            if sidecar is not None:
                sidecar.flush()
            store.delete_path(path)
            store.add(rows)
            if motion_store is not None:
                motion_store.delete_path(path)
                motion_store.add(motion_rows)
            manifest.mark_done(path, duration, settings)
            counts["indexed"] += 1
            counts["segments"] += len(rows)
            note = f" ({unreadable} unreadable skipped)" if unreadable else ""
            log(f"    {len(rows)} segments in {time.time() - t_start:.1f}s{note}")
        except Exception as e:
            manifest.mark_error(path, f"{type(e).__name__}: {e}")
            counts["errors"] += 1
            log(f"    ERROR: {e}")
    if counts["segments"]:
        store.rebuild_fts()
    return counts


def caption_indexed(
    manifest: Manifest,
    store: SegmentStore,
    caption_backend: CaptionBackend,
    folder,
    frame_w: int = 640,
    frame_h: int = 360,
    caption_dir=None,
    caption_frames: int | None = None,
    caption_frames_short: int | None = None,
    caption_frames_long: int | None = None,
    short_video_s: float = SHORT_VIDEO_S,
    long_video_s: float = LONG_VIDEO_S,
    force: bool = False,
    log=print,
) -> dict:
    """Caption segments already in the store, without touching their vectors.

    Captioning and embedding change on different cadences -- a new caption
    model, prompt or raster should not cost a re-index -- so this reads the
    spans indexing recorded and decodes only the stills it sends. That also
    frees the raster: `index --caption` can only show the model the 456x256
    frames decoded for the embedding.

    A segment with a sidecar caption is not bought again unless `force`; its
    cached caption is still written into the row, so an index built without
    --caption picks up captions bought earlier.
    """
    caption_backend.preflight()
    prefix = str(Path(folder).resolve()).rstrip(os.sep) + os.sep
    by_path = defaultdict(list)
    for r in store.segments():
        if r["path"].startswith(prefix):
            by_path[r["path"]].append(r)
    counts = {"videos": 0, "captioned": 0, "reused": 0, "errors": 0}
    for i, (path, rows) in enumerate(sorted(by_path.items()), 1):
        log(f"[{i}/{len(by_path)}] {path} ({len(rows)} segments)")
        recorded = manifest.recorded_settings(path)
        stride = recorded["stride"] if recorded else DEFAULT_STRIDE
        per_window = (recorded["frames_per_window"] if recorded
                      else DEFAULT_FRAMES_PER_WINDOW)
        # The last window always ends at the video's end (`windows`).
        n_frames = frames_for_duration(
            max(r["t1"] for r in rows), short=caption_frames_short,
            normal=caption_frames, long=caption_frames_long,
            short_video_s=short_video_s, long_video_s=long_video_s)
        sidecar = CaptionSidecar(path, caption_dir)
        sidecar.check_writable()
        try:
            for r in sorted(rows, key=lambda r: r["t0"]):
                t0, t1 = r["t0"], r["t1"]
                caption = None if force else sidecar.get(r["id"])
                if caption is None:
                    stills = extract_stills(
                        path, still_times(t0, t1, stride, n_frames, per_window),
                        frame_w, frame_h)
                    if len(stills) == 0:
                        log(f"    {r['id']}: no frames decoded, skipped")
                        continue
                    caption = caption_backend.caption(stills, t1 - t0, n_frames, stride)
                    sidecar.put(r["id"], t0, t1, caption, caption_backend.name)
                    counts["captioned"] += 1
                else:
                    counts["reused"] += 1
                text = f"{manifest.transcript_between(r['video_id'], t0, t1)} {caption}".strip()
                if text != r["text"]:
                    store.set_text(r["id"], text)
            counts["videos"] += 1
        except Exception as e:
            counts["errors"] += 1
            log(f"    ERROR: {e}")
        finally:
            # Captions already paid for are kept even when a later one fails.
            sidecar.flush()
    if counts["captioned"] or counts["reused"]:
        store.rebuild_fts()
    return counts
