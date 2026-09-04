import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from .ffbin import ffmpeg_exe
from .probe import keyframe_before, probe


def export_clip(
    path: str,
    t0: float,
    t1: float,
    out: str,
    pad: float = 1.0,
    snap: bool = True,
    smart: bool = True,
) -> tuple[str, float, float]:
    """Export [t0, t1] as a clip. Returns (out, actual_t0, actual_t1).

    Default (smart=True) is smart_cut: frame-accurate at both ends, with
    only the boundary GOPs recoded and the interior passed through
    bit-identically.

    smart=False falls back to a pure stream copy, which cannot start
    anywhere but a keyframe: with snap=True the start moves back to the
    previous keyframe, so the file is byte-identical to the source but
    may begin up to one GOP early. Use it when the extra footage does not
    matter and speed does, or where the smartcut extra is unavailable.
    """
    if smart:
        return smart_cut(path, t0, t1, out, pad=pad)
    duration = probe(path)["duration"]
    start = max(0.0, t0 - pad)
    end = min(duration, t1 + pad)
    if snap:
        start = keyframe_before(path, start)
    # Camera files may carry data streams (e.g. Canon timed metadata,
    # codec "none") that cannot be muxed back into MP4; fall back to
    # video+audio only if copying everything fails.
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


def smart_cut(
    path: str, t0: float, t1: float, out: str, pad: float = 1.0
) -> tuple[str, float, float]:
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

    duration = probe(path)["duration"]
    start = max(0.0, t0 - pad)
    end = min(duration, t1 + pad)
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


def write_llc_projects(
    hits: list[dict], out_dir: str | Path | None = None, label: str = ""
) -> list[str]:
    """One LosslessCut project (.llc, JSON) per video, hits as cut segments.

    LosslessCut resolves media as path.join(dirname(project), mediaFileName)
    with no sanitization, so relative mediaFileName paths work. Default:
    projects go in an `llc/` subfolder next to each video (mediaFileName
    "../<video>"), keeping the video folders uncluttered. With out_dir,
    mediaFileName is the relative path from out_dir to the video — open the
    files in place, do not move them.
    """
    by_video: dict[str, list[dict]] = defaultdict(list)
    for h in hits:
        by_video[h["path"]].append(h)
    written: list[str] = []
    for path, video_hits in by_video.items():
        video = Path(path)
        if out_dir is None:
            target_dir = video.parent / "llc"
            media_ref = f"../{video.name}"
        else:
            target_dir = Path(out_dir)
            media_ref = os.path.relpath(path, target_dir)
        project = {
            "version": 1,
            "mediaFileName": media_ref,
            "cutSegments": [
                {"start": h["t0"], "end": h["t1"], "name": label}
                for h in sorted(video_hits, key=lambda h: h["t0"])
            ],
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / f"{video.stem}.llc"
        n = 1
        while str(out) in written:
            n += 1
            out = target_dir / f"{video.stem}_{n}.llc"
        out.write_text(json.dumps(project, indent=2))
        written.append(str(out))
    return written
