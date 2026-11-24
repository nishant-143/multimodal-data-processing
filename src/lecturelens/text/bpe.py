"""Byte-Pair Encoding (Sennrich et al., 2016) trained from scratch on a word-frequency corpus."""
from __future__ import annotations

from collections import Counter

END = "</w>"


class BPE:
    def __init__(self, merges: list[tuple[str, str]] | None = None):
        self.merges = merges or []
        self.ranks = {m: i for i, m in enumerate(self.merges)}

    @classmethod
    def train(cls, words: list[str], num_merges: int = 500, min_freq: int = 2) -> "BPE":
        vocab = Counter(tuple(w) + (END,) for w in words)
        merges = []
        for _ in range(num_merges):
            pairs: Counter = Counter()
            for sym, f in vocab.items():
                for a, b in zip(sym, sym[1:]):
                    pairs[a, b] += f
            if not pairs:
                break
            best, freq = pairs.most_common(1)[0]
            if freq < min_freq:
                break
            merges.append(best)
            new_vocab = Counter()
            for sym, f in vocab.items():
                new_vocab[_merge(sym, best)] += f
            vocab = new_vocab
        return cls(merges)

    def encode_word(self, word: str) -> list[str]:
        sym = tuple(word) + (END,)
        while len(sym) > 1:
            pairs = [(self.ranks.get(p, 1 << 30), p) for p in zip(sym, sym[1:])]
            rank, best = min(pairs)
            if rank == 1 << 30:
                break
            sym = _merge(sym, best)
        return [s.replace(END, "") for s in sym if s.replace(END, "")]

    def encode(self, text: str) -> list[str]:
        return [p for w in text.split() for p in self.encode_word(w)]


def _merge(sym: tuple, pair: tuple) -> tuple:
    out, i = [], 0
    while i < len(sym):
        if i < len(sym) - 1 and (sym[i], sym[i + 1]) == pair:
            out.append(sym[i] + sym[i + 1])
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return tuple(out)
