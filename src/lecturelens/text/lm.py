"""N-gram language models with Laplace, add-k and interpolated Kneser–Ney smoothing (bigram)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class BigramLM:
    def __init__(self, sentences: list[list[str]], smoothing: str = "kn", k: float = 0.1,
                 discount: float = 0.75, min_count: int = 1):
        counts = Counter(w for s in sentences for w in s)
        self.vocab = {w for w, c in counts.items() if c >= min_count} | {EOS, UNK}
        self.smoothing, self.k, self.d = smoothing, k, discount
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        for s in sentences:
            toks = [BOS] + [self._m(w) for w in s] + [EOS]
            self.uni.update(toks[:-1])           # context counts
            self.bi.update(zip(toks, toks[1:]))
        self.V = len(self.vocab)
        # Kneser–Ney statistics
        self.followers = defaultdict(set)        # w_prev -> {w}
        self.preceders = defaultdict(set)        # w -> {w_prev}
        for (a, b) in self.bi:
            self.followers[a].add(b)
            self.preceders[b].add(a)
        self.n_bigram_types = len(self.bi)

    def _m(self, w: str) -> str:
        return w if w in self.vocab else UNK

    def prob(self, w: str, prev: str) -> float:
        w, c_prev = self._m(w), self.uni[prev]
        c = self.bi[(prev, w)]
        if self.smoothing == "mle":
            return c / c_prev if c_prev else 0.0
        if self.smoothing == "laplace":
            return (c + 1) / (c_prev + self.V)
        if self.smoothing == "addk":
            return (c + self.k) / (c_prev + self.k * self.V)
        # interpolated Kneser–Ney
        p_cont = (len(self.preceders[w]) + 1) / (self.n_bigram_types + self.V)  # +1: unseen words
        if c_prev == 0:
            return p_cont
        lam = self.d * len(self.followers[prev]) / c_prev
        return max(c - self.d, 0) / c_prev + lam * p_cont

    def logprob(self, sent: list[str]) -> tuple[float, int]:
        toks = [BOS] + [self._m(w) for w in sent] + [EOS]
        lp = 0.0
        for a, b in zip(toks, toks[1:]):
            p = self.prob(b, a)
            lp += math.log(p) if p > 0 else float("-inf")
        return lp, len(toks) - 1

    def perplexity(self, sentences: list[list[str]]) -> float:
        lp, n = 0.0, 0
        for s in sentences:
            a, b = self.logprob(s)
            lp, n = lp + a, n + b
        return math.exp(-lp / max(n, 1))
