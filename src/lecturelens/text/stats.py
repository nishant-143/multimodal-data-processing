"""Corpus statistics: Zipf's law (rank-frequency) and Heaps' law (vocabulary growth)."""
from __future__ import annotations

from collections import Counter

import numpy as np


def zipf(tokens: list[str], top: int | None = None):
    """Returns ranks, freqs and the fitted exponent s in f(r) ~ C * r^-s (log-log least squares)."""
    freqs = np.array(sorted(Counter(tokens).values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    n = top or len(freqs)
    # Fit on ranks with freq >= 2 to avoid the hapax "staircase" tail dominating the fit
    m = (freqs[:n] >= 2)
    slope, intercept = np.polyfit(np.log(ranks[:n][m]), np.log(freqs[:n][m]), 1)
    return ranks, freqs, -slope, np.exp(intercept)


def heaps(tokens: list[str], points: int = 50):
    """Vocabulary size V(n) after n tokens and fitted V = K * n^beta."""
    seen, V = set(), []
    for t in tokens:
        seen.add(t)
        V.append(len(seen))
    V = np.array(V, dtype=float)
    n = np.unique(np.linspace(1, len(tokens), points).astype(int))
    beta, logK = np.polyfit(np.log(n), np.log(V[n - 1]), 1)
    return n, V[n - 1], np.exp(logK), beta


def type_token_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / max(len(tokens), 1)
