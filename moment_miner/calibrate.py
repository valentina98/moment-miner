"""Per-machine index speed, measured once, so `mm index` can print an ETA.

Index time is almost all per-segment work (decode 8 frames, embed them), so
one rate -- seconds per segment -- predicts a run from its segment count.
The rate is measured by indexing a generated 4K clip, the resolution the
footage is shot at, and stored in the data dir keyed by machine and backend:
a data dir moved to another machine gets a fresh measurement there.
"""

import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

from .ffbin import ffmpeg_exe
from .indexer import index_pending, windows
from .manifest import Manifest
from .probe import probe
from .store import SegmentStore

CLIP = {"size": "3840x2160", "rate": 25, "seconds": 32}


def machine_key() -> str:
    """CPU model, core count and GPU: what the rate depends on.

    Not the hostname -- under Docker that is a fresh container id every run.
    """
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return f"{cpu} x{os.cpu_count()} / {gpu}"


def _path(data_dir) -> Path:
    return Path(data_dir) / "calibration.json"


def load_rate(data_dir, backend_name: str) -> dict | None:
    try:
        entries = json.loads(_path(data_dir).read_text())
    except (OSError, ValueError):
        return None
    return entries.get(f"{machine_key()} / {backend_name}")


def calibrate(data_dir, backend, log=print, clip: dict = CLIP) -> dict:
    """Index a generated clip with `backend` and store seconds per segment."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "clip"
        folder.mkdir()
        out = folder / "calibration.mp4"
        subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
             f"testsrc2=size={clip['size']}:rate={clip['rate']}"
             f":duration={clip['seconds']}",
             "-c:v", "libx264", "-preset", "ultrafast", str(out)],
            check=True, capture_output=True,
        )
        manifest = Manifest(Path(tmp) / "manifest.db")
        manifest.scan(folder)
        t = time.time()
        result = index_pending(manifest, SegmentStore(Path(tmp) / "data", backend.name),
                               backend, use_asr=False, log=lambda *_: None)
        elapsed = time.time() - t
    if not result["segments"]:
        raise RuntimeError(f"calibration clip produced no segments: {result}")
    entry = {
        "s_per_segment": round(elapsed / result["segments"], 2),
        "segments": result["segments"],
        "clip": f"{clip['size']} {clip['rate']} fps {clip['seconds']} s",
        "measured_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    path = _path(data_dir)
    try:
        entries = json.loads(path.read_text())
    except (OSError, ValueError):
        entries = {}
    entries[f"{machine_key()} / {backend.name}"] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n")
    log(f"calibrated: {entry['s_per_segment']} s per segment "
        f"({machine_key()}, {backend.name})")
    return entry


def estimate(rows, rate: dict, win: float, stride: float) -> tuple[int, float, float]:
    """(segments, video seconds, estimated seconds) for the videos in `rows`."""
    segments, seconds = 0, 0.0
    for row in rows:
        duration = probe(row["path"])["duration"]
        seconds += duration
        segments += len(list(windows(duration, win, stride)))
    return segments, seconds, segments * rate["s_per_segment"]
