import time

from .embeddings.base import EmbeddingBackend
from .frames import extract_frames
from .manifest import Manifest
from .probe import probe
from .store import SegmentStore


def windows(duration: float, win: float = 8.0, stride: float = 4.0):
    t = 0.0
    while t < duration:
        yield t, min(t + win, duration)
        t += stride


def index_pending(
    manifest: Manifest,
    store: SegmentStore,
    backend: EmbeddingBackend,
    use_asr: bool = True,
    asr_model: str = "small",
    win: float = 8.0,
    stride: float = 4.0,
    frames_per_window: int = 8,
    log=print,
) -> dict:
    counts = {"indexed": 0, "errors": 0, "segments": 0}
    videos = manifest.pending()
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
            rows = []
            unreadable = 0
            for t0, t1 in windows(duration, win, stride):
                try:
                    frames = extract_frames(path, t0, t1, n=frames_per_window)
                except Exception:
                    # A truncated file fails only from the window past its last
                    # packet on; everything before it is still worth indexing.
                    unreadable += 1
                    continue
                if len(frames) == 0:
                    continue
                rows.append({
                    "id": f"{vid}:{t0:.1f}",
                    "video_id": vid,
                    "path": path,
                    "t0": t0,
                    "t1": t1,
                    "text": manifest.transcript_between(vid, t0, t1),
                    "vector": backend.embed_segment(frames),
                })
            if not rows:
                raise RuntimeError(f"no readable windows ({unreadable} failed)")
            store.delete_path(path)
            store.add(rows)
            manifest.mark_done(path, duration, backend.name)
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
