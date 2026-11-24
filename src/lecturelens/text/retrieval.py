"""Vector-space retrieval written from scratch: TF-IDF + cosine, and Okapi BM25."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


class TfidfIndex:
    """Sublinear TF (1+log tf) x smoothed IDF (log((1+N)/(1+df))+1), L2-normalised rows."""

    def __init__(self, docs: list[list[str]], vocab: list[str] | None = None):
        self.N = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.vocab = vocab or sorted(df)
        self.tid = {t: i for i, t in enumerate(self.vocab)}
        self.idf = np.array([math.log((1 + self.N) / (1 + df.get(t, 0))) + 1 for t in self.vocab])
        self.matrix = np.vstack([self.vectorize(d) for d in docs]) if docs else np.zeros((0, len(self.vocab)))

    def vectorize(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, c in Counter(toks).items():
            if t in self.tid:
                v[self.tid[t]] = 1 + math.log(c)
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n else v

    def search(self, query: list[str]) -> np.ndarray:
        return self.matrix @ self.vectorize(query)


class BM25:
    """Okapi BM25: idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl)), idf = ln((N-df+.5)/(df+.5)+1)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.len.mean() if len(docs) else 0.0
        N = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.len / max(self.avgdl, 1e-9))
        for t in query:
            if t not in self.idf:
                continue
            tf = np.array([d.get(t, 0) for d in self.docs], dtype=float)
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return s


def reciprocal_rank(ranked_ids: list, relevant: set) -> float:
    for i, d in enumerate(ranked_ids, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0
