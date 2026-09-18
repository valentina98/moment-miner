"""Caption backends: one sentence of searchable text per segment.

Captions are appended to the `text` column that the transcript already fills
(store.py:7), which carries the FTS index (store.py:52) that search.py fuses
with vision ranking. Most of this footage is silent, so without captions the
text half of the fusion contributes nothing.

Captions are indexed content, never ground truth: a caption written by a model
and scored against a label written by the same model measures agreement, not
accuracy. See docs/captions.md.
"""

import base64
import csv
import io
import json
import os
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import numpy as np

# Frames are sampled evenly across the whole window, in time order, so the
# model can say what changes rather than describe one instant.
#
# The span reaches both ends deliberately. Windows overlap by half, so every
# interior second is covered twice and an inset would cost little there — but
# the last window's end is the end of the video, covered by nothing else, and
# in this footage that is exactly where the payoff is: the trick lands and
# recording stops. An earlier 0.15-0.85 span never sampled the final eighth of
# any window, so the last ~1 s of every clip was invisible to captioning.
SAMPLE_FIRST, SAMPLE_LAST = 0.0, 1.0
DEFAULT_FRAMES = 3
DEFAULT_SHORT_FRAMES = 1
SHORT_VIDEO_S = 8.0
LONG_VIDEO_S = 120.0
FRAME_WIDTH = 640

PROMPT = """These {n} images are one {span:.0f}-second video segment, sampled across it in time order.

Write ONE label of 5 to 12 words. A label is what someone would type into a search box to find this clip again — not a description of the picture.

- Lead with the action or event, then the setting: "kong vault over a rail, concrete plaza".
- Plain, common words. The word a person would search, not the more precise one: "bar", not "horizontal calisthenics apparatus".
- No sentences, no framing ("a video showing", "the image shows"), no commentary on the shot itself.
- Name only what you can see. If the subject is too small or unclear to identify, write "unclear subject" and then only what is certain.
- If nothing happens, say what the scene is: "empty skatepark at dusk".

Reply with the label alone, no preamble and no quotes."""


class MissingCaptionCredentials(RuntimeError):
    """Neither credential route is available."""


class CaptionBackend(ABC):
    """Turns one segment's frames into one line of searchable text."""

    name: str

    @abstractmethod
    def caption(self, frames: np.ndarray, span: float,
                n_frames: int | None = None, stride: float = 4.0) -> str:
        """(k, h, w, 3) uint8 frames of one segment -> one sentence."""

    def preflight(self) -> None:
        """Resolve credentials before indexing starts.

        Without this a missing key surfaces on the first segment, inside the
        per-video error handler, which marks every video `error` in the
        manifest for what is a config mistake.
        """


class MockCaptionBackend(CaptionBackend):
    """Deterministic, no network. The backend tests and `make test` use this."""

    name = "mock"

    def caption(self, frames: np.ndarray, span: float,
                n_frames: int | None = None, stride: float = 4.0) -> str:
        picked = subsample(frames, span, stride, n_frames)
        return f"mock caption of {len(picked)} frames over {span:.1f}s"


def sample_points(span: float, stride: float) -> list[float]:
    """Where in a segment of `span` seconds to take stills, as fractions of it.

    Three tiers, keyed on the stride rather than on a frame count:

    - shorter than one stride: one frame, in the middle. Nothing else fits.
    - shorter than two strides: two frames, at a third and two thirds.
    - otherwise: anchored half a stride from the end and stepping back a whole
      stride at a time, as many as fit strictly inside the span. The anchor is
      at the end because that is where the action resolves in this footage --
      the trick lands and recording stops.

    A whole stride, not half of one, so the result is exactly one frame per
    stride-length block of video, taken at that block's midpoint. Windows
    overlap by half, so neighbouring windows then agree on the frames they
    share rather than interleaving new ones between them. At the defaults
    (8 s window, 4 s stride) a full segment yields two frames, at 2 s and 6 s.

    **Revisit this if you change --stride.** The tiers and the half-stride step
    were reasoned about at stride 4; at stride 2 a full 8 s window would yield
    seven frames instead of three, and frames are the dominant cost of a
    caption pass. Nothing here adapts to that on its own.
    """
    if span < stride:
        return [0.5]
    if span < 2 * stride:
        return [1 / 3, 2 / 3]
    points, t = [], span - stride / 2
    while t > 0:
        points.append(t / span)
        t -= stride
    return sorted(points)


def even_points(n: int) -> list[float]:
    """n positions spread across the whole window, used when a count is forced."""
    if n <= 1:
        return [SAMPLE_LAST]
    gap = (SAMPLE_LAST - SAMPLE_FIRST) / (n - 1)
    return [SAMPLE_FIRST + i * gap for i in range(n)]


def subsample(frames: np.ndarray, span: float = 8.0, stride: float = 4.0,
              n: int | None = None) -> np.ndarray:
    """Pick the stills to caption out of those decoded for the embedding.

    By default the count follows from the segment's span and the stride
    (`sample_points`). An explicit n overrides that and spreads n frames evenly
    across the whole window instead.
    """
    points = even_points(n) if n else sample_points(span, stride)
    if len(frames) <= len(points):
        return frames
    idx = sorted({min(len(frames) - 1, int(p * len(frames))) for p in points})
    return frames[idx]


def frames_for_duration(
    duration: float,
    *,
    short: int | None = None,
    normal: int | None = None,
    long: int | None = None,
    short_video_s: float = SHORT_VIDEO_S,
    long_video_s: float = LONG_VIDEO_S,
) -> int | None:
    """A forced frame count for a video of this length, or None to derive one.

    None is the normal answer: `sample_points` derives the count from the
    segment's span and the stride, and that is the rule. These tiers exist only
    so a count can be forced per video length.
    """
    if duration <= short_video_s:
        return short
    if duration >= long_video_s and long is not None:
        return long
    return normal


def _encode(frame: np.ndarray) -> str:
    from PIL import Image

    img = Image.fromarray(frame)
    if img.width > FRAME_WIDTH:
        img = img.resize((FRAME_WIDTH, round(img.height * FRAME_WIDTH / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _find_token(obj) -> str | None:
    """Pull an OAuth access token out of the Claude Code credential file.

    Walks the JSON rather than hard-coding a path into it: the file is written
    by Claude Code, its shape is not part of any published contract, and this
    must never read or log anything but the one value it needs.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.lower().replace("_", "") == "accesstoken":
                return v
            found = _find_token(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_token(v)
            if found:
                return found
    return None


def resolve_credentials(env=None, config_dir=None, *, allow_paid: bool = False,
                       paid_budget_usd: float | None = None) -> dict:
    """The free route, or the paid one only when it was asked for and capped.

    Returns kwargs for anthropic.Anthropic(). The subscription token spends
    session quota; ANTHROPIC_API_KEY spends money. So the token wins whenever it
    exists, and the key is used only when the caller both allows it and states a
    positive budget -- Valya's rule, 2026-09-18: the paid route stays available
    but is never reached by default. Until that day the order was reversed, and
    a key exported for any other reason was enough to turn a caption pass paid;
    the Makefile passes `-e ANTHROPIC_API_KEY` straight into the container.

    The budget is a gate, not a meter: nothing here counts dollars as the pass
    runs. Metering is a separate piece of work, recorded as a next step.
    """
    env = os.environ if env is None else env
    base = Path(config_dir or env.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude"))
    cred = base / ".credentials.json"
    if cred.exists():
        token = _find_token(json.loads(cred.read_text()))
        if token:
            return {"auth_token": token}
    key = env.get("ANTHROPIC_API_KEY")
    if key:
        if allow_paid and paid_budget_usd is not None and paid_budget_usd > 0:
            return {"api_key": key}
        raise MissingCaptionCredentials(
            "ANTHROPIC_API_KEY is set but the paid route is refused: it needs "
            "--allow-paid AND --paid-budget-usd above 0 (got "
            f"allow_paid={allow_paid!r}, paid_budget_usd={paid_budget_usd!r}). "
            f"The free route is Claude Code credentials at {cred} -- sign in, or "
            "mount that file into the container."
        )
    raise MissingCaptionCredentials(
        f"no captioning credentials: sign in to Claude Code so that {cred} "
        "exists (mount it into the container). ANTHROPIC_API_KEY is the paid "
        "alternative and needs --allow-paid with a budget. Captioning is the "
        "only stage that needs either; indexing without --caption needs neither."
    )


class ClaudeCaptionBackend(CaptionBackend):
    """Hosted Claude. One request per segment, three stills per request."""

    def __init__(self, model: str, max_tokens: int = 200, client=None, env=None,
                 allow_paid: bool = False, paid_budget_usd: float | None = None):
        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self._env = env
        self._allow_paid = allow_paid
        self._paid_budget_usd = paid_budget_usd

    @property
    def client(self):
        if self._client is None:
            import anthropic

            kwargs = resolve_credentials(
                env=self._env, allow_paid=self._allow_paid,
                paid_budget_usd=self._paid_budget_usd)
            if "auth_token" in kwargs:
                kwargs["default_headers"] = {"anthropic-beta": "oauth-2025-04-20"}
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def preflight(self) -> None:
        self.client

    def caption(self, frames: np.ndarray, span: float,
                n_frames: int | None = None, stride: float = 4.0) -> str:
        picked = subsample(frames, span, stride, n_frames)
        content = [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode(f)}}
            for f in picked
        ]
        content.append({"type": "text", "text": PROMPT.format(n=len(picked), span=span)})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return " ".join(
            b.text.strip() for b in response.content if b.type == "text"
        ).strip()


def get_caption_backend(name: str, **kwargs) -> CaptionBackend:
    if name == "mock":
        return MockCaptionBackend()
    if name.startswith("claude-"):
        return ClaudeCaptionBackend(name, **kwargs)
    raise ValueError(
        f"unknown caption backend: {name!r} (use a Claude model id such as "
        "claude-haiku-4-5, or 'mock')"
    )


SIDECAR_NAME = "captions.csv"
SIDECAR_COLUMNS = ["id", "t0", "t1", "caption", "model", "written"]


class CaptionSidecar:
    """Captions on disk beside the footage they describe.

    `<folder>/captions.csv`, the same place and shape as the `labels.csv` that
    `mm eval` reads — but captions are machine-written indexed content and
    labels are human-checked ground truth, so they stay separate files.

    A caption spends session quota (money, on the API-key route) and an index
    does not survive a deleted mm_data volume, so captions are written next to
    the video and read back on the next pass: re-indexing the same folder
    re-uses them instead of spending for them again.
    """

    def __init__(self, video_path: str | Path, caption_dir: str | Path | None = None):
        base = Path(caption_dir) if caption_dir else Path(video_path).parent
        self.path = base / SIDECAR_NAME
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(newline="") as f:
                self.rows = {r["id"]: r for r in csv.DictReader(f)}
        self._dirty = False

    def get(self, seg_id: str) -> str | None:
        row = self.rows.get(seg_id)
        return row["caption"] if row else None

    def put(self, seg_id: str, t0: float, t1: float, caption: str, model: str) -> None:
        self.rows[seg_id] = {
            "id": seg_id, "t0": f"{t0:.1f}", "t1": f"{t1:.1f}",
            "caption": caption, "model": model, "written": date.today().isoformat(),
        }
        self._dirty = True

    def check_writable(self) -> None:
        """Fail before the first caption request, not after.

        The Makefile mounts the archive read-only, which is right for every
        other command and wrong for this one — `make caption` mounts it rw.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            probe = self.path.parent / ".write-test"
            probe.touch()
            probe.unlink()
        except OSError as e:
            raise RuntimeError(
                f"cannot write captions beside the footage ({self.path.parent}): {e}. "
                "The archive is mounted read-only — use `make caption`, which mounts "
                "it read-write, or pass --caption-dir."
            ) from e

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SIDECAR_COLUMNS)
            w.writeheader()
            for row in sorted(self.rows.values(), key=lambda r: float(r["t0"])):
                w.writerow(row)
        self._dirty = False
