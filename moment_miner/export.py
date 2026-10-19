import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import click

from .ffbin import ffmpeg_exe
from .guard import writable
from .hostpaths import on_host
from .probe import keyframe_before, probe


class NameTaken(click.ClickException):
    def __init__(self, path: str):
        super().__init__(f"not written: {path} already exists")
        self.path = path


def _lc_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    s, ms = divmod(ms, 1000)
    return f"{s // 3600:02d}.{s % 3600 // 60:02d}.{s % 60:02d}.{ms:03d}"


def clip_name(path: str | Path, start: float, end: float, prefix: str = "") -> str:
    """LosslessCut's default export name, `<video>-<from>-<to><ext>`, with the
    exact cut times, so a clip names the span it holds."""
    video = Path(path)
    return f"{prefix}{video.stem}-{_lc_time(start)}-{_lc_time(end)}{video.suffix}"


def cut_span(path: str, t0: float, t1: float, pad: float = 1.0, snap: bool = True,
             smart: bool = True) -> tuple[float, float]:
    duration = probe(path)["duration"]
    start = max(0.0, t0 - pad)
    end = min(duration, t1 + pad)
    if snap and not smart:
        start = keyframe_before(path, start)
    return start, end


def export_clip(
    path: str,
    t0: float,
    t1: float,
    out: str,
    pad: float = 1.0,
    snap: bool = True,
    smart: bool = True,
    prefix: str = "",
) -> tuple[str, float, float]:
    """Export [t0, t1] as a clip. Returns (out, actual_t0, actual_t1).

    `out` is the file to create, or a folder to create it in under
    `prefix` + `clip_name`. An existing file is never replaced: NameTaken.

    Default (smart=True) is smart_cut: frame-accurate at both ends, with
    only the boundary GOPs recoded and the interior passed through
    bit-identically.

    smart=False falls back to a pure stream copy, which cannot start
    anywhere but a keyframe: with snap=True the start moves back to the
    previous keyframe, so the file is byte-identical to the source but
    may begin up to one GOP early. Use it when the extra footage does not
    matter and speed does, or where the smartcut extra is unavailable.
    """
    start, end = cut_span(path, t0, t1, pad, snap, smart)
    if Path(out).is_dir():
        out = str(Path(out) / clip_name(path, start, end, prefix))
    if Path(out).exists():
        raise NameTaken(out)
    writable(out)
    if smart:
        return smart_cut(path, start, end, out)
    # Camera files may carry data streams (e.g. Canon timed metadata,
    # codec "none") that cannot be muxed back into MP4; fall back to
    # video+audio only if copying everything fails. `-y` only lets the
    # second try replace the first one's failed file: the name was free.
    errors = []
    for mapping in (["-map", "0", "-ignore_unknown"], ["-map", "0:v", "-map", "0:a?"]):
        result = subprocess.run(
            [ffmpeg_exe(), "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", path,
             "-t", f"{end - start:.3f}", *mapping, "-c", "copy",
             "-avoid_negative_ts", "make_zero", out],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 1024:
            return out, start, end
        errors.append(result.stderr.strip())
    raise RuntimeError("ffmpeg export failed: " + " | ".join(errors))


def smart_cut(path: str, start: float, end: float, out: str) -> tuple[str, float, float]:
    """Frame-accurate export via the `smartcut` package (optional extra).

    Stream copy can only begin on a keyframe, so a plain copy snaps
    backwards and hands back up to one GOP of footage nobody asked for
    (0.48 s on the Canon sources here). smartcut recodes only the GOPs
    it has to at both ends and passes the rest through untouched: the
    copied interior decodes bit-identically to the source.

    Pinned deliberately. The package was deprecated in February 2026, but
    it is MIT, pure Python, and its only heavy dependency is PyAV; if a
    future PyAV breaks it, vendoring the four modules used here is the
    escape hatch. Do not replace it with hand-rolled ffmpeg calls — that
    is what this code used to be, and it cut four frames early on real
    50 fps footage while passing its own tests.
    """
    try:
        from fractions import Fraction

        from smartcut.cut_video import (AudioExportInfo, AudioExportSettings,
                                        VideoExportMode, VideoExportQuality,
                                        VideoSettings, smart_cut as _smart_cut)
        from smartcut.media_container import MediaContainer
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError(
            "smart cut needs the 'smartcut' extra: pip install -e '.[smartcut]'"
        ) from exc

    container = MediaContainer(path)
    try:
        audio = [AudioExportSettings(codec="passthru")] * len(container.audio_tracks)
        # smart_cut returns the exception instead of raising it.
        failure = _smart_cut(
            container,
            [(Fraction(start).limit_denominator(1000),
              Fraction(end).limit_denominator(1000))],
            out,
            audio_export_info=AudioExportInfo(output_tracks=audio),
            video_settings=VideoSettings(VideoExportMode.SMARTCUT,
                                         VideoExportQuality.NEAR_LOSSLESS, "copy"),
            log_level="error",
        )
    finally:
        container.close()
    if failure is not None:
        raise RuntimeError(f"smart cut failed: {failure}") from failure
    return out, start, end


def write_llc_projects(hits: list[dict], out_dir: str | Path,
                       label: str = "") -> tuple[list[str], list[str]]:
    """One LosslessCut project (.llc, JSON) per video, hits as cut segments.
    Returns (written, taken): a project whose file already exists is skipped.

    LosslessCut resolves media as path.join(dirname(project), mediaFileName)
    with no sanitization, so mediaFileName is the relative path from out_dir
    to the video — open the files in place, do not move them. Inside a
    container it is computed between the host paths, see `on_host`.
    """
    target_dir = writable(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # A hit is the search's guess, so its name says so: "kong vault? #2".
    by_video: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for rank, h in enumerate(hits, 1):
        by_video[h["path"]].append((rank, h))
    written: list[str] = []
    taken: list[str] = []
    for path, video_hits in by_video.items():
        video = Path(path)
        project = {
            "version": 1,
            "mediaFileName": os.path.relpath(on_host(path), on_host(target_dir)),
            "cutSegments": [
                {"start": h["t0"], "end": h["t1"], "name": f"{label}? #{rank}"}
                for rank, h in sorted(video_hits, key=lambda rh: rh[1]["t0"])
            ],
        }
        out = target_dir / f"{video.stem}.llc"
        n = 1
        while str(out) in written:
            n += 1
            out = target_dir / f"{video.stem}_{n}.llc"
        if out.exists():
            taken.append(str(out))
            continue
        out.write_text(json.dumps(project, indent=2))
        written.append(str(out))
    return written, taken
