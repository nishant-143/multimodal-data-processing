"""End-to-end LectureLens pipeline.

    uv run python -m lecturelens.run                       # all lectures in data/raw
    uv run python -m lecturelens.run --only lec01          # one lecture

For each lecture writes to data/processed/<name>/:
    speech.npz        VAD/energy/pitch/MFCC-flux tracks and pauses
    slides.json       visual segments (type, deck page, OCR + PDF text)
    timeline.parquet  per-10 s window features from every modality (+ unsupervised importance)
    insights.json     off-slide insights
    smart_speed.wav   pause-trimmed + 1.3x WSOLA audio
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .fusion.align import agg_series, count_events, text_in_windows, window_edges
from .fusion.importance import unsupervised_score
from .ingest import ingest, load_audio
from .speech.features import vad
from .speech.ola import smart_speed
from .speech.pitch import pitch_track
from .speech.spectral import mfcc
from .text.asr import transcribe, words_of
from .text.lecture_text import boilerplate_mask, corpus_idf, off_slide_insights, window_text_features
from .text.lm import BigramLM
from .text.normalize import content_tokens, tokenize
from .vision.deck import DeckMatcher, load_deck
from .vision.lecture_vision import process_frames, slide_crop
from .vision.ocr import ocr
from .vision.slides import detect_changes, shot_aware_scores

RAW, PROC, SLIDES = Path("data/raw"), Path("data/processed"), Path("data/slides")
WIN = 10.0


def deck_for(name: str) -> Path | None:
    m = re.match(r"lec(\d+)", name)
    p = SLIDES / f"lec{int(m.group(1))}.pdf" if m else None
    return p if p and p.exists() else None


# ---- speech ----------------------------------------------------------------------------------------

def speech_tracks(d: Path, force=False) -> dict:
    p = d / "speech.npz"
    if p.exists() and not force:
        z = np.load(p, allow_pickle=True)
        return {k: z[k] for k in z.files}
    x, sr = load_audio(d)
    v = vad(x, sr)
    tp, f0 = pitch_track(x, sr, method="autocorr")
    M = mfcc(x, sr)
    flux = np.r_[0, np.linalg.norm(np.diff(M, axis=1), axis=0)]
    tm = np.arange(M.shape[1]) * 160 / sr
    out = dict(t=v.times, energy_db=v.energy_db, zcr=v.zcr, speech=v.speech, pauses=np.array(v.pauses),
               t_pitch=tp, f0=f0, t_mfcc=tm, mfcc_flux=flux)
    np.savez_compressed(p, **out)
    return out


# ---- vision ----------------------------------------------------------------------------------------

def visual_segments(d: Path, fs, deck: Path | None, force=False) -> list[dict]:
    p = d / "slides.json"
    if p.exists() and not force:
        return json.loads(p.read_text())
    z = shot_aware_scores(fs.dists, fs.types)
    changes = detect_changes(z, fs.times, k=1.0, already_z=True, min_gap_s=3.0)
    bounds = [0.0] + changes + [float(fs.times[-1]) + 0.5]
    matcher = DeckMatcher(load_deck(deck)) if deck else None
    pages = matcher.pages if matcher else []
    segs = []
    import cv2
    for s, e in zip(bounds, bounds[1:]):
        idx = [i for i, t in enumerate(fs.times) if s <= t < e]
        if not idx:
            continue
        types = [fs.types[i] for i in idx]
        kind = max(set(types), key=types.count)
        seg = {"start": s, "end": e, "type": kind, "page": None, "match_sim": None, "pdf_text": "", "ocr": ""}
        if kind == "slide":
            mid = [i for i in idx if fs.types[i] == "slide"][len([i for i in idx if fs.types[i] == "slide"]) // 2]
            g = cv2.imread(str(sorted((d / "frames").glob("*.jpg"))[mid]), cv2.IMREAD_GRAYSCALE)
            crop = slide_crop(g)
            seg["ocr"] = " ".join(ocr(crop).split())
            seg["frame"] = mid
            if matcher:
                pg, sim, margin = matcher.match(crop)
                seg.update(page=pg, match_sim=sim, match_margin=margin, pdf_text=pages[pg].text)
        segs.append(seg)
    # slide ids: same deck page => same slide (revisits), else per segment
    for i, sg in enumerate(segs):
        sg["slide_id"] = f"p{sg['page']}" if sg["page"] is not None else f"s{i}"
    p.write_text(json.dumps(segs, indent=1))
    return segs


# ---- text ------------------------------------------------------------------------------------------

def build_corpus_models(names: list[str]):
    """IDF over 30 s transcript chunks of all lectures; bigram LM per held-out lecture."""
    chunks, sents = [], {}
    for n in names:
        words = words_of(json.loads((PROC / n / "transcript.json").read_text()))
        for t0 in np.arange(0, words[-1]["end"], 30):
            chunks.append(content_tokens(" ".join(w["word"] for w in words if t0 <= w["start"] < t0 + 30)))
        segs = json.loads((PROC / n / "transcript.json").read_text())
        sents[n] = [tokenize(s["text"]) for s in segs]
    return corpus_idf(chunks), sents


# ---- per-lecture assembly ----------------------------------------------------------------------------

def build_timeline(n: str, idf, sents) -> pd.DataFrame:
    d = PROC / n
    meta = json.loads((d / "meta.json").read_text())
    sp = speech_tracks(d)
    fs = pickle.load(open(d / "frames.pkl", "rb"))
    segs = visual_segments(d, fs, deck_for(n))
    words = words_of(json.loads((d / "transcript.json").read_text()))
    E = window_edges(meta["duration"], WIN)
    nW = len(E) - 1
    df = pd.DataFrame({"lecture": n, "t0": E[:-1], "t1": E[1:]})

    # speech
    sp_mask = sp["speech"].astype(bool)
    df["speech_ratio"] = agg_series(sp["t"], sp_mask.astype(float), E)
    df["energy_db"] = agg_series(sp["t"][sp_mask], sp["energy_db"][sp_mask], E)
    df["energy_std"] = agg_series(sp["t"][sp_mask], sp["energy_db"][sp_mask], E, np.nanstd)
    voiced = sp["f0"] > 0
    df["pitch_mean"] = agg_series(sp["t_pitch"][voiced], sp["f0"][voiced], E)
    df["pitch_range"] = agg_series(sp["t_pitch"][voiced], sp["f0"][voiced], E,
                                   lambda v: np.percentile(v, 90) - np.percentile(v, 10))
    pauses = sp["pauses"] if len(sp["pauses"]) else np.zeros((0, 2))
    df["n_pauses"] = count_events(list(pauses[:, 1]), E)
    df["pause_before"] = agg_series(pauses[:, 1], pauses[:, 1] - pauses[:, 0], E, np.max) if len(pauses) else np.nan
    df["pause_before"] = df["pause_before"].fillna(0)
    n_words = count_events([w["start"] for w in words], E)
    df["speak_rate"] = n_words / np.maximum(df["speech_ratio"].fillna(0) * WIN, 1.0)
    df["mfcc_flux"] = agg_series(sp["t_mfcc"], sp["mfcc_flux"], E)

    # vision
    change_t = [s["start"] for s in segs[1:]]
    df["visual_change"] = count_events(change_t, E)
    df["diff_energy"] = agg_series(fs.times, fs.diff_energy, E)
    df["flow_mag"] = agg_series(fs.times, fs.flow_mag, E)
    df["is_slide"] = agg_series(fs.times, np.array([t == "slide" for t in fs.types], float), E)
    df["text_lines"] = agg_series(fs.times, fs.text_lines, E)

    # text
    wtexts = text_in_windows(words, E)
    stexts = [" ".join((s["pdf_text"] or s["ocr"]) for s in segs if s["start"] < e and s["end"] > b)
              for b, e in zip(E[:-1], E[1:])]
    train = [s for k, v in sents.items() if k != n for s in v]          # LM never sees this lecture
    lm = BigramLM(train, smoothing="kn", min_count=2)
    for k, v in window_text_features(wtexts, stexts, idf, lm).items():
        df[k] = v
    df["text"] = wtexts
    df["slide_text"] = stexts
    others = [" ".join(w["word"] for w in words_of(json.loads((PROC / k / "transcript.json").read_text())))
              for k in sents if k != n]
    df["boilerplate"] = boilerplate_mask(wtexts, others)
    feats = df.drop(columns=["lecture", "t0", "t1", "text", "slide_text", "boilerplate"])
    imp = unsupervised_score(feats[~df.boilerplate])
    df["importance"] = imp.reindex(df.index).fillna(imp.min())
    df.to_parquet(d / "timeline.parquet")

    slide_segs = [dict(s, ocr=(s["pdf_text"] or s["ocr"])) for s in segs if s["type"] == "slide"]
    ins = off_slide_insights(words, slide_segs, idf)
    (d / "insights.json").write_text(json.dumps(ins, indent=1))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--smart-speed", action="store_true")
    a = ap.parse_args()
    names = sorted(v.stem for v in RAW.glob("*.mp4"))
    for n in names:
        d = PROC / n
        ingest(RAW / f"{n}.mp4", d)
        transcribe(d / "audio.wav", d / "transcript.json")
        if not (d / "frames.pkl").exists():
            pickle.dump(process_frames(d / "frames"), open(d / "frames.pkl", "wb"))
    idf, sents = build_corpus_models(names)
    for n in names:
        if a.only and not n.startswith(a.only):
            continue
        df = build_timeline(n, idf, sents)
        print(f"{n}: {len(df)} windows, hand-weighted top moments (run scripts/eval_fusion.py for learned ones):",
              ", ".join(f"{int(t // 60)}:{int(t % 60):02d}" for t in df.nlargest(5, "importance")["t0"]))
        if a.smart_speed:
            x, sr = load_audio(PROC / n)
            sp = speech_tracks(PROC / n)
            y = smart_speed(x, sr, [tuple(p) for p in sp["pauses"]], rate=1.3)
            sf.write(PROC / n / "smart_speed.wav", y.astype(np.float32), sr)
            print(f"  smart-speed: {len(x) / sr / 60:.1f} min -> {len(y) / sr / 60:.1f} min")


if __name__ == "__main__":
    main()
