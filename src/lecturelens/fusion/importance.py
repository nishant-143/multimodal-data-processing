"""Conventional multimodal learning: unimodal vs. early fusion vs. late fusion, leave-one-lecture-out."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPEECH = ["speech_ratio", "energy_db", "energy_std", "pitch_mean", "pitch_range", "pause_before",
          "n_pauses", "speak_rate", "mfcc_flux"]
VISION = ["visual_change", "diff_energy", "flow_mag", "is_slide", "text_lines"]
TEXT = ["salience", "perplexity", "slide_overlap", "novelty", "n_content"]
MODALITIES = {"speech": SPEECH, "vision": VISION, "text": TEXT}


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def unsupervised_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Label-free emphasis score: mean of signed z-features per modality, then averaged (late fusion).

    Signs encode the hypothesis: louder, higher/more varied pitch, a pause before, slower speech,
    a fresh slide, more motion, salient/novel/off-slide words => more important.
    """
    signs = {"energy_db": 1, "pitch_mean": 1, "pitch_range": 1, "pause_before": 1, "speak_rate": -1,
             "mfcc_flux": 1, "visual_change": 1, "flow_mag": 1, "salience": 1, "novelty": 1,
             "slide_overlap": -1, "perplexity": 1}
    weights = weights or {"speech": 1.0, "vision": 1.0, "text": 1.0}
    z = (df - df.mean()) / df.std(ddof=0).replace(0, 1)
    parts = []
    for m, cols in MODALITIES.items():
        cols = [c for c in cols if c in signs and c in z]
        parts.append(weights[m] * sum(signs[c] * z[c].fillna(0) for c in cols) / len(cols))
    return sum(parts) / sum(weights.values())


def evaluate(df: pd.DataFrame, label: str = "label", group: str = "lecture",
             modalities: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Leave-one-lecture-out. Returns AP and F1 for each unimodal model, early and late fusion."""
    modalities = modalities or MODALITIES
    rows = []
    feats = {**modalities, "early": [c for cols in modalities.values() for c in cols]}
    for held in df[group].unique():
        tr, te = df[df[group] != held], df[df[group] == held]
        if te[label].nunique() < 2:
            continue
        probs = {}
        for name, cols in feats.items():
            m = _model().fit(tr[cols].fillna(0), tr[label])
            probs[name] = m.predict_proba(te[cols].fillna(0))[:, 1]
        probs["late"] = np.mean([probs[m] for m in modalities], axis=0)
        probs["unsupervised"] = _minmax(unsupervised_score(te[SPEECH + VISION + TEXT]).to_numpy())
        for name, p in probs.items():
            thr = 0.5 if name != "unsupervised" else np.quantile(p, 1 - te[label].mean())
            rows.append({"held_out": held, "model": name,
                         "AP": average_precision_score(te[label], p),
                         "F1": f1_score(te[label], p >= thr),
                         "base_rate": te[label].mean()})
    return pd.DataFrame(rows)


def coefficients(df: pd.DataFrame, label: str = "label", cols: list[str] | None = None) -> pd.Series:
    cols = cols or SPEECH + VISION + TEXT
    m = _model().fit(df[cols].fillna(0), df[label])
    return pd.Series(m[-1].coef_[0], index=cols).sort_values()


def _minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)
