"""Rank segments for a query by reading their text, not their pixels.

The index already ranks two ways — vector similarity over frame embeddings and
BM25 over the caption and transcript text. A classifier is a third: it reads
one segment's text and answers how well it matches the query, which is a
judgement neither of the others makes. Whether that judgement is worth its cost
is what `mm probe` exists to measure, so these stay behind an interface and
nothing in `search` depends on them.
"""

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
    QUESTION = ("Does this video segment description show {query}? "
                "Answer about the description alone.")

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
        questions = {"match": {"type": "noul",
                               "instructions": self.QUESTION.format(query=query)}}
        out = []
        for text in texts:
            answer = self.router.predict({"description": text}, questions)
            out.append(float(answer["answers"]["match"]["probability"]))
        return out


def ranker_names() -> list[str]:
    return sorted(RANKERS)


RANKERS = {"mock": MockRanker, "laya": LayaRanker}


def get_ranker(name: str) -> Ranker:
    try:
        return RANKERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown ranker: {name!r} (available: {', '.join(ranker_names())})"
        ) from None
