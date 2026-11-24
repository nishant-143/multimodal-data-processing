"""Put every modality on one timeline of fixed windows.

Modalities arrive at different rates — audio frames every 10 ms, video samples every 1 s, words at
irregular ASR timestamps, slide segments of arbitrary length — so each is aggregated into W-second
windows. Windows where a modality is absent (no slide on screen, silence) get a mask flag instead of
being dropped, so fusion can handle missing modalities explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def window_edges(duration: float, win: float = 10.0) -> np.ndarray:
    return np.arange(0, duration + win, win)


def agg_series(t: np.ndarray, v: np.ndarray, edges: np.ndarray, fn=np.nanmean) -> np.ndarray:
    idx = np.digitize(t, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(out)):
        sel = v[idx == i]
        sel = sel[~np.isnan(sel)] if sel.dtype.kind == "f" else sel
        if len(sel):
            out[i] = fn(sel)
    return out


def count_events(times: list[float], edges: np.ndarray) -> np.ndarray:
    return np.histogram(times, bins=edges)[0].astype(float)


def text_in_windows(words: list[dict], edges: np.ndarray) -> list[str]:
    buckets = [[] for _ in range(len(edges) - 1)]
    for w in words:
        i = int(np.searchsorted(edges, (w["start"] + w["end"]) / 2, side="right") - 1)
        if 0 <= i < len(buckets):
            buckets[i].append(w["word"])
    return [" ".join(b) for b in buckets]


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Per-lecture standardisation, so features are comparable across speakers/recordings."""
    return (df - df.mean()) / df.std(ddof=0).replace(0, 1)
