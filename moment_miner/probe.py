import json
import subprocess

from .ffbin import ffprobe_exe


def probe(path: str) -> dict:
    out = subprocess.run(
        [ffprobe_exe(), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise ValueError(f"no video stream: {path}")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    return {
        "duration": float(info["format"].get("duration") or v.get("duration") or 0),
        "width": v.get("width"),
        "height": v.get("height"),
        "codec": v.get("codec_name"),
        "has_audio": a is not None,
        "pix_fmt": v.get("pix_fmt"),
        "fps": _ratio(v.get("r_frame_rate")),
        "audio_codec": (a or {}).get("codec_name"),
        "audio_rate": (a or {}).get("sample_rate"),
        "audio_channels": (a or {}).get("channels"),
    }


def _ratio(value: str | None) -> float:
    """ffprobe rationals ("25/1"); 0.0 when absent or degenerate."""
    if not value:
        return 0.0
    num, _, den = value.partition("/")
    try:
        d = float(den) if den else 1.0
        return float(num) / d if d else 0.0
    except ValueError:
        return 0.0


def _keyframe_pts(path: str, start: float, end: float) -> list[float]:
    """Keyframe timestamps in [start, end], from packet flags.

    Reads packet flags (no decoding); frame-level pts_time is empty on
    some ffprobe builds.
    """
    out = subprocess.run(
        [ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0",
         "-read_intervals", f"{start}%{end}", path],
        capture_output=True, text=True, check=True,
    ).stdout
    pts = []
    for line in out.splitlines():
        value, _, flags = line.partition(",")
        try:
            parsed = float(value)
        except ValueError:
            continue
        if "K" in flags:
            pts.append(parsed)
    return sorted(pts)


def keyframe_before(path: str, t: float, lookback: float = 30.0) -> float:
    """Latest keyframe timestamp <= t (for lossless stream-copy cuts).

    Widens the search window once for long-GOP files before giving up
    and returning 0.0.
    """
    for lb in (lookback, lookback * 10):
        start = max(0.0, t - lb)
        candidates = [p for p in _keyframe_pts(path, start, t + 0.5) if p <= t + 1e-3]
        if candidates:
            return max(candidates)
        if start == 0.0:
            break
    return 0.0


def keyframe_after(path: str, t: float, lookahead: float = 30.0) -> float | None:
    """Earliest keyframe strictly after t; None when the file has none left."""
    for la in (lookahead, lookahead * 10):
        candidates = [p for p in _keyframe_pts(path, max(0.0, t - 0.1), t + la)
                      if p > t + 1e-3]
        if candidates:
            return min(candidates)
    return None
