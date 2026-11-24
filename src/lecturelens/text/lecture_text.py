"""Lecture-level text features built on the from-scratch primitives:

* chunked BM25 / TF-IDF "ask the lecture" search over transcript + slide text
* slide <-> speech alignment and off-slide insights (vector-space difference)
* per-window salience, novelty and LM perplexity
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lm import BigramLM
from .normalize import content_tokens, tokenize
from .retrieval import BM25, TfidfIndex


@dataclass
class Chunk:
    start: float
    end: float
    text: str           # spoken text
    slide_text: str     # OCR text of slide(s) on screen during the chunk


def make_chunks(words: list[dict], slide_segs: list[dict], length: float = 30.0, hop: float = 15.0,
                duration: float | None = None) -> list[Chunk]:
    duration = duration or (words[-1]["end"] if words else 0)
    out = []
    t = 0.0
    while t < duration:
        s, e = t, min(t + length, duration)
        spoken = " ".join(w["word"] for w in words if s <= w["start"] < e)
        slides = " ".join(sg["ocr"] for sg in slide_segs if sg["start"] < e and sg["end"] > s and sg.get("ocr"))
        out.append(Chunk(s, e, spoken, slides))
        t += hop
    return out


class LectureSearch:
    """BM25 over chunks; slide text is included with a lower weight (appended once vs. spoken text)."""

    def __init__(self, chunks: list[Chunk], use_slides: bool = True, method: str = "bm25"):
        self.chunks = chunks
        docs = [content_tokens(c.text + (" " + c.slide_text if use_slides else "")) for c in chunks]
        self.method = method
        self.index = BM25(docs) if method == "bm25" else TfidfIndex(docs)

    def search(self, query: str, k: int = 5) -> list[tuple[float, Chunk]]:
        q = content_tokens(query)
        s = self.index.scores(q) if self.method == "bm25" else self.index.search(q)
        order = np.argsort(-s)[:k]
        return [(float(s[i]), self.chunks[i]) for i in order if s[i] > 0]


def off_slide_insights(words: list[dict], slide_segs: list[dict], idf: dict[str, float],
                       sent_len: float = 20.0, top: int = 10) -> list[dict]:
    """Spoken passages that are salient (high mean IDF) yet dissimilar from the slide on screen."""
    out = []
    for sg in slide_segs:
        if not sg.get("ocr") or len(content_tokens(sg["ocr"])) < 3:
            continue
        seg_words = [w for w in words if sg["start"] <= w["start"] < sg["end"]]
        t = sg["start"]
        while t < sg["end"]:
            ws = [w for w in seg_words if t <= w["start"] < t + sent_len]
            t += sent_len / 2
            toks = content_tokens(" ".join(w["word"] for w in ws))
            if len(toks) < 6:
                continue
            idx = TfidfIndex([toks, content_tokens(sg["ocr"])], vocab=sorted(set(toks) | set(content_tokens(sg["ocr"]))))
            idx.idf = np.array([idf.get(v, max(idf.values(), default=1.0)) for v in idx.vocab])
            a, b = idx.vectorize(toks), idx.vectorize(content_tokens(sg["ocr"]))
            cos = float(a @ b)
            sal = float(np.mean([idf.get(w, max(idf.values(), default=1.0)) for w in toks]))
            out.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": " ".join(w["word"] for w in ws),
                        "slide_id": sg["slide_id"], "slide_cos": cos, "salience": sal,
                        "score": sal * (1 - cos)})
    out.sort(key=lambda d: -d["score"])
    # de-duplicate overlapping passages
    picked: list[dict] = []
    for d in out:
        if all(d["end"] <= p["start"] or d["start"] >= p["end"] for p in picked):
            picked.append(d)
        if len(picked) == top:
            break
    return picked


def corpus_idf(docs: list[list[str]]) -> dict[str, float]:
    N = len(docs)
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((1 + N) / (1 + n)) + 1 for t, n in df.items()}


def shingles(text: str, n: int = 5) -> set[tuple]:
    t = tokenize(text)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def boilerplate_mask(window_texts: list[str], other_lectures: list[str], n: int = 5, thresh: float = 0.5) -> np.ndarray:
    """Data cleaning: a window is boilerplate if >= thresh of its word 5-grams also occur in another
    lecture's transcript (e.g. the OpenCourseWare licence notice read at the start of every video)."""
    other = set().union(*(shingles(t, n) for t in other_lectures)) if other_lectures else set()
    out = []
    for wt in window_texts:
        sh = shingles(wt, n)
        out.append(bool(sh) and len(sh & other) / len(sh) >= thresh)
    return np.array(out)


import re as _re

# Discourse cues (classical lexical patterns): how lecturers mark definitions, rules and emphasis.
CUES = {
    "cue_define": r"\b(is called|are called|we call|is defined|defined as|definition|is a way|means that|"
                  r"refers to|stands for|is an? (?:\w+ )?(?:object|function|method|class|way|thing|set|sequence))\b",
    "cue_rule": r"\b(always|never|must|have to|has to|the rule|in general|in other words|so the idea|key idea|"
                r"the point is|the way to think)\b",
    "cue_flag": r"\b(important|remember|careful|notice|crucial|key thing|keep in mind|the big idea|"
                r"common mistake|this is the)\b",
}


def cue_counts(text: str) -> dict[str, int]:
    t = text.lower()
    return {k: len(_re.findall(p, t)) for k, p in CUES.items()}


def window_text_features(window_texts: list[str], slide_texts: list[str], idf: dict[str, float],
                         lm: BigramLM | None, shrink_m: float = 8.0) -> dict[str, np.ndarray]:
    """Per-window text features. Novelty is the number of new content-word *types* in the window divided
    by the number Heaps' law (V = K n^beta, fitted on this lecture) predicts at that point, so the start
    of a lecture, where nearly every word is new, is not automatically 'novel'."""
    from .stats import heaps
    max_idf = max(idf.values(), default=1.0)
    all_toks = [t for wt in window_texts for t in content_tokens(wt)]
    K, beta = (heaps(all_toks)[2:] if len(all_toks) > 50 else (1.0, 1.0))
    n_seen = 0
    seen: set[str] = set()
    feats = {k: [] for k in ["salience", "perplexity", "slide_overlap", "novelty", "n_content", *CUES]}
    for wt, st in zip(window_texts, slide_texts):
        for k, v in cue_counts(wt).items():
            feats[k].append(v)
        toks = content_tokens(wt)
        stoks = set(content_tokens(st))
        feats["n_content"].append(len(toks))
        feats["salience"].append(np.mean([idf.get(t, max_idf) for t in toks]) if toks else np.nan)
        feats["slide_overlap"].append(len(set(toks) & stoks) / len(set(toks)) if toks and stoks else np.nan)
        new_types = len(set(toks) - seen)
        expected = K * ((n_seen + len(toks)) ** beta - max(n_seen, 1) ** beta) if toks else 0
        feats["novelty"].append(new_types / max(expected, 1e-6) if toks else np.nan)
        seen.update(toks)
        n_seen += len(toks)
        raw = tokenize(wt)
        feats["perplexity"].append(math.log(lm.perplexity([raw])) if lm and len(raw) >= 3 else np.nan)
    out = {k: np.array(v, float) for k, v in feats.items()}
    # Shrink averages computed from few words toward the lecture mean (m pseudo-words): a window with
    # three content words, one of them a rare name, should not out-rank a dense explanation.
    n = out["n_content"]
    for k in ("salience", "novelty", "slide_overlap"):
        mu = np.nanmean(out[k])
        out[k] = np.where(np.isnan(out[k]), np.nan, (n * np.nan_to_num(out[k]) + shrink_m * mu) / (n + shrink_m))
    return out
