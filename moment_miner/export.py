import json
import os
import re
from collections import defaultdict
from pathlib import Path

import click
import json5

from . import ffbin
from .ffbin import ffmpeg_exe
from .guard import WriteRefused, writable
from .hostpaths import in_container, on_host
from .manifest import PARTIAL
from .probe import keyframe_before, probe


class NameTaken(click.ClickException):
    def __init__(self, path: str):
        super().__init__(f"not written: {on_host(path)} already exists")
        self.path = path


def slug(text: str) -> str:
    """`kong vault?` -> `kong-vault`, safe in a file name on every system."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "query"


def _lc_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    s, ms = divmod(ms, 1000)
    return f"{s // 3600:02d}.{s % 3600 // 60:02d}.{s % 60:02d}.{ms:03d}"


def clip_name(path: str | Path, start: float, end: float, prefix: str = "") -> str:
    """LosslessCut's default export name, `<video>-<from>-<to><ext>`, with the
    exact cut times, so a clip names the span it holds."""
    video = Path(path)
    return f"{prefix}{video.stem}-{_lc_time(start)}-{_lc_time(end)}{video.suffix}"


def export_clip(
    path: str,
    start: float,
    end: float,
    out: str,
    smart: bool = True,
    prefix: str = "",
) -> tuple[str, float, float]:
    """Cut [start, end] of `path` into a clip, exactly; no margin is added.
    Returns (out, actual_start, actual_end).

    `out` is the file to create, or a folder to create it in under
    `prefix` + `clip_name`. An existing file is never replaced: NameTaken.

    Default (smart=True) is smart_cut: frame-accurate at both ends, with
    only the boundary GOPs recoded and the interior passed through
    bit-identically.

    smart=False falls back to a pure stream copy, which cannot start
    anywhere but a keyframe: the start moves back to the previous keyframe,
    so the file is byte-identical to the source but may begin up to one GOP
    early. Use it when the extra footage does not matter and speed does, or
    where the smartcut extra is unavailable.
    """
    if not smart:
        start = keyframe_before(path, start)
    if Path(out).is_dir():
        out = str(Path(out) / clip_name(path, start, end, prefix))
    if Path(out).exists():
        raise NameTaken(out)
    writable(out)
    # Written under a hidden name and renamed once complete, so a crash, a
    # kill or Ctrl+C never leaves a truncated clip that looks like a result.
    part = str(Path(out).with_name(PARTIAL + Path(out).name))
    try:
        if smart:
            smart_cut(path, start, end, part)
        else:
            _stream_copy(path, start, end, part)
        if Path(out).exists():
            raise NameTaken(out)
        os.rename(part, out)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return out, start, end


def _stream_copy(path: str, start: float, end: float, out: str) -> None:
    # Camera files may carry data streams (e.g. Canon timed metadata,
    # codec "none") that cannot be muxed back into MP4; fall back to
    # video+audio only if copying everything fails. `-y` only lets the
    # second try replace the first one's failed file.
    errors = []
    for mapping in (["-map", "0", "-ignore_unknown"], ["-map", "0:v", "-map", "0:a?"]):
        result = ffbin.run_while_growing(
            [ffmpeg_exe(), "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", path,
             "-t", f"{end - start:.3f}", *mapping, "-c", "copy",
             "-avoid_negative_ts", "make_zero", out],
            out, text=True,
        )
        if result.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 1024:
            return
        errors.append(result.stderr.strip())
    raise RuntimeError("ffmpeg export failed: " + " | ".join(errors))


def smart_cut(path: str, start: float, end: float, out: str) -> None:
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


def write_llc_projects(hits: list[dict], out_dir: str | Path,
                       query: str) -> tuple[list[str], list[str]]:
    """One LosslessCut project per video, `<video>-<query>.llc`, its hits as
    cut segments, in rank order of the list given. The spans are written as
    given, so what LosslessCut shows is what `mm cut` cuts. Returns
    (written, taken): a project whose file already exists is skipped.

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
            "version": 2,
            "mediaFileName": os.path.relpath(on_host(path), on_host(target_dir)),
            "cutSegments": [
                {"start": h["t0"], "end": h["t1"], "name": f"{query}? #{rank}"}
                for rank, h in sorted(video_hits, key=lambda rh: rh[1]["t0"])
            ],
        }
        name = f"{video.stem}-{slug(query)}"
        out = target_dir / f"{name}.llc"
        n = 1
        while str(out) in written:
            n += 1
            out = target_dir / f"{name}_{n}.llc"
        if out.exists():
            taken.append(str(out))
            continue
        out.write_text(json.dumps(project, indent=2))
        written.append(str(out))
    return written, taken


RANK = re.compile(r"#(\d+)\s*$")


def read_project(project: str | Path) -> tuple[str, list[tuple[float, float | None, str]]]:
    """A LosslessCut project's video and its segments as (start, end, name).

    LosslessCut saves JSON5 and drops an end that runs to the video's end;
    segments it would not export (`selected: false`) are left out here too.
    The video is resolved the way LosslessCut does, from the project's folder
    on the host, then mapped back into the container.
    """
    data = json5.loads(Path(project).read_text())
    if not data.get("mediaFileName"):
        raise click.ClickException(f"{on_host(project)}: names no video (mediaFileName)")
    video = in_container(os.path.normpath(
        os.path.join(os.path.dirname(on_host(project)), data["mediaFileName"])))
    segments = [(float(s.get("start", 0.0)), s.get("end"), s.get("name", ""))
                for s in data.get("cutSegments", []) if s.get("selected", True)]
    return video, segments


def cut_project(project: str | Path, out_dir: str | Path, smart: bool = True):
    """Cut every segment of a project into `out_dir`, exactly as it is.

    A segment whose name ends in a rank (`kong vault? #2`) gets it in front,
    `02_<video>-<from>-<to><ext>`, so a folder sorts best match first. A
    taken name or a failed cut is recorded and the next segment still runs.
    Returns (written, taken, failed): written as (clip, video, start, end).
    """
    video, segments = read_project(project)
    if not os.path.isfile(video):
        raise click.ClickException(
            f"{on_host(project)}: its video {on_host(video)} is not there")
    duration = probe(video)["duration"]
    target = writable(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written, taken, failed = [], [], []
    for start, end, name in segments:
        end = duration if end is None else min(float(end), duration)
        rank = RANK.search(name)
        prefix = f"{int(rank.group(1)):02d}_" if rank else ""
        try:
            if end <= start:
                raise ValueError(f"segment ends at {end:.3f} s, before its start")
            out, a0, a1 = export_clip(video, start, end, str(target), smart=smart,
                                      prefix=prefix)
            written.append((out, video, a0, a1))
        except NameTaken as e:
            taken.append(e.path)
        except WriteRefused:
            raise
        except Exception as e:
            failed.append(f"{on_host(video)} {start:.3f}-{end:.3f}: {e}")
    return written, taken, failed
