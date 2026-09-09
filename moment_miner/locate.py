"""Score a video frame by frame against a query, then pick the region that answers it.

The index stores one mean-pooled vector per segment, which is enough to rank clips but
says nothing about *where* inside one the action is. Every frame vector needed for that
already exists at index time and is averaged away (`embeddings/base.py` pools them), so
locating is a matter of not discarding them rather than of finer segmentation.

Locators are interchangeable on purpose: extent is content-dependent -- one pull-up or
ten, three corks or fifteen -- so the default infers it from the curve, while a caller
who knows the length asks for it instead.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    t0: float
    t1: float
    score: float
    peak: float


def heatmap(frame_vecs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """(k, dim) frames and one (dim,) query, both L2-normalized → (k,) cosine per frame."""
    return np.asarray(frame_vecs, dtype=np.float32) @ np.asarray(query_vec, dtype=np.float32)


def _period(times: np.ndarray) -> float:
    return float(np.median(np.diff(times))) if len(times) > 1 else 0.0


def _span(times: np.ndarray, i: int, j: int) -> tuple[float, float]:
    """Frames i..j inclusive → the wall-clock span they cover."""
    return float(times[i]), float(times[j]) + _period(times)


class RegionLocator(ABC):
    """Turns a per-frame score curve into the spans worth cutting."""

    name: str

    @abstractmethod
    def locate(self, scores: np.ndarray, times: np.ndarray) -> list[Region]:
        """Scores and their timestamps, both length k → regions, best first."""


class MaxSubarray(RegionLocator):
    """Longest contiguous run that beats the clip's own baseline (Kadane).

    Mean-over-window always prefers the single best frame and sum-over-window always
    prefers the whole clip; centering on a baseline makes above-baseline frames pay in
    and below-baseline frames pay out, so the run ends where the action does and its
    length is never asserted. For a repetition query the length *is* the answer -- no
    single frame distinguishes three corks from fifteen.
    """

    name = "maxsub"

    def __init__(self, baseline: str | float = "mean", top_k: int = 1):
        if not (baseline in ("median", "mean") or
                (isinstance(baseline, (int, float)) and 0 < baseline < 100)):
            raise ValueError(
                f"unknown baseline {baseline!r}; use 'mean', 'median', or a percentile")
        self.baseline = baseline
        self.top_k = top_k

    def _baseline(self, scores: np.ndarray) -> float:
        """Must sit strictly above the clip's floor, which is why the default is the mean.

        Most of a clip is not the action, so on a flat floor the median lands *on* it:
        every floor frame then centers to exactly 0, which neither extends a run nor
        ends one, and Kadane merges two separate bumps across the dead stretch between
        them. The mean is pulled above the floor by the bump itself. It fails the other
        way when the action fills most of the clip -- pass a percentile for that.
        """
        if self.baseline == "mean":
            return float(np.mean(scores))
        if self.baseline == "median":
            return float(np.median(scores))
        return float(np.percentile(scores, self.baseline))

    def locate(self, scores: np.ndarray, times: np.ndarray) -> list[Region]:
        scores = np.asarray(scores, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if scores.size == 0:
            return []
        centered = scores - self._baseline(scores)
        out: list[Region] = []
        for _ in range(self.top_k):
            i, j = _kadane(centered)
            if i is None:
                break
            t0, t1 = _span(times, i, j)
            out.append(Region(t0, t1, float(centered[i:j + 1].sum()),
                              float(scores[i:j + 1].max())))
            centered[i:j + 1] = -np.inf
            if not np.isfinite(centered).any():
                break
        return out


class FixedWindow(RegionLocator):
    """Highest-scoring window of a caller-supplied length.

    For when the length is known and should not be inferred -- a fixed-duration cut, or
    a query whose extent the user is asserting rather than discovering.
    """

    name = "window"

    def __init__(self, duration: float, top_k: int = 1):
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.duration = duration
        self.top_k = top_k

    def locate(self, scores: np.ndarray, times: np.ndarray) -> list[Region]:
        scores = np.asarray(scores, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if scores.size == 0:
            return []
        period = _period(times) or self.duration
        width = max(1, min(len(scores), int(round(self.duration / period))))
        means = np.convolve(scores, np.ones(width) / width, mode="valid")
        out: list[Region] = []
        for _ in range(self.top_k):
            if not np.isfinite(means).any():
                break
            i = int(np.nanargmax(means))
            j = i + width - 1
            t0, t1 = _span(times, i, j)
            out.append(Region(t0, t1, float(means[i]), float(scores[i:j + 1].max())))
            means[max(0, i - width + 1):i + width] = -np.inf
        return out


def _kadane(a: np.ndarray) -> tuple[int | None, int | None]:
    """Indices of the maximum-sum contiguous subarray, inclusive.

    An all-negative curve returns its single best element rather than the empty run,
    which is the wanted answer here: the best frame is still the best frame.
    """
    best = cur = -np.inf
    best_i = best_j = start = None
    for k, x in enumerate(a):
        if not np.isfinite(x):
            cur = -np.inf
            start = None
            continue
        if start is None or cur <= 0:
            cur, start = x, k
        else:
            cur += x
        if cur > best:
            best, best_i, best_j = cur, start, k
    return best_i, best_j


LOCATORS = {"maxsub": MaxSubarray, "window": FixedWindow}


def build_locator(name: str, duration: float | None = None, top_k: int = 1) -> RegionLocator:
    """`window` needs a duration; `maxsub` refuses one, since inferring it is the point."""
    if name not in LOCATORS:
        raise ValueError(f"unknown locator {name!r}; available: {sorted(LOCATORS)}")
    if name == "window":
        if duration is None:
            raise ValueError("locator 'window' needs --duration")
        return FixedWindow(duration, top_k=top_k)
    if duration is not None:
        raise ValueError("locator 'maxsub' infers the duration; use --locator window to set one")
    return MaxSubarray(top_k=top_k)
