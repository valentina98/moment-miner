"""Rank segments for a query by reading their text, not their pixels.

The index already ranks two ways — vector similarity over frame embeddings and
BM25 over the caption and transcript text. A classifier is a third: it reads
one segment's text and answers how well it matches the query, which is a
judgement neither of the others makes. Whether that judgement is worth its cost
is what `mm probe` exists to measure, so these stay behind an interface and
nothing in `search` depends on them.
"""

import json
import os
import urllib.request
from abc import ABC, abstractmethod


class Ranker(ABC):
    """Scores one query against many texts. Higher is a better match."""

    name: str

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """One score per text, in the order given."""

    def top(self, query: str, rows: list[dict], k: int = 5,
            axis: str = "caption") -> list[dict]:
        """Rank `rows` for `query`, reading one axis of each row.

        The axis is the whole point: a query constrains what it names, and the
        clip-level axes repeat across every window of a clip, so scoring the
        joined caption drowns the one part that varies.
        """
        scores = self.score(query, [r.get(axis) or "" for r in rows])
        ranked = sorted(zip(rows, scores), key=lambda p: p[1], reverse=True)
        return [dict(r, score=s) for r, s in ranked[:k]]


class MockRanker(Ranker):
    """Word overlap. Deterministic, dependency-free, and the tests' baseline."""

    name = "mock"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(query.lower().split())
        out = []
        for t in texts:
            words = set(t.lower().split())
            out.append(len(q & words) / len(q) if q else 0.0)
        return out


class LayaRanker(Ranker):
    """Laya, a 421M non-autoregressive decision model, run locally on CPU.

    It answers a typed question about one text with a calibrated probability,
    which is the same shape a hosted classifier returns — so the comparison
    `mm probe` draws is between judgements, not between architectures.
    """

    name = "laya"
    QUESTION = ("The text describes what happens in a video segment. "
                "It shows {query}.")

    def __init__(self, model: str = "convaiinnovations/laya", router=None):
        self.model = model
        self._router = router

    @property
    def router(self):
        if self._router is None:
            from laya import Router

            self._router = Router(preload=True)
        return self._router

    def score(self, query: str, texts: list[str]) -> list[float]:
        return self.score_many([query], texts)[query]

    def score_many(self, queries: list[str], texts: list[str]
                   ) -> dict[str, list[float]]:
        """Every query against every text, one forward pass per text.

        Laya resolves all of a request's questions in a single pass, so asking
        ten prompts at once costs about what one costs — a full corpus sweep is
        minutes rather than an hour.
        """
        keys = {f"q{i}": q for i, q in enumerate(queries)}
        questions = {k: {"type": "noul",
                         "instructions": self.QUESTION.format(query=q)}
                     for k, q in keys.items()}
        out: dict[str, list[float]] = {q: [] for q in queries}
        for text in texts:
            answers = self.router.predict({"description": text}, questions)["answers"]
            for k, q in keys.items():
                out[q].append(float(answers[k]["noul"]))
        return out


class JevRanker(Ranker):
    """Jev, a hosted decision model, over the same typed questions Laya takes.

    Hosted and metered, so it is offered as an instrument and never wired into
    `search`: a paid dependency in the product is a separate decision from a
    paid dependency in a measurement.
    """

    name = "jev"
    URL = "https://api.typesafe.ai/v1/systemone"
    QUESTION = LayaRanker.QUESTION

    def __init__(self, api_key: str | None = None):
        self._key = api_key or os.environ.get("TYPESAFE_API_KEY")

    def score(self, query: str, texts: list[str]) -> list[float]:
        return self.score_many([query], texts)[query]

    def score_many(self, queries: list[str], texts: list[str]
                   ) -> dict[str, list[float]]:
        if not self._key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set, and the jev ranker is metered: "
                "export it, or use --ranker laya, which runs locally.")
        keys = {f"q{i}": q for i, q in enumerate(queries)}
        questions = {k: {"type": "noul",
                         "instructions": self.QUESTION.format(query=q)}
                     for k, q in keys.items()}
        out: dict[str, list[float]] = {q: [] for q in queries}
        for text in texts:
            answers = self._ask(text, questions)
            for k, q in keys.items():
                out[q].append(float(answers.get(k, {}).get("noul", 0.0)))
        return out

    def _ask(self, state: str, questions: dict) -> dict:
        body = json.dumps({"state": state, "questions": questions}).encode()
        req = urllib.request.Request(
            self.URL, data=body,
            headers={"Authorization": f"Bearer {self._key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r).get("answers", {})


def ranker_names() -> list[str]:
    return sorted(RANKERS)


RANKERS = {"mock": MockRanker, "laya": LayaRanker, "jev": JevRanker}


def get_ranker(name: str) -> Ranker:
    try:
        return RANKERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown ranker: {name!r} (available: {', '.join(ranker_names())})"
        ) from None
