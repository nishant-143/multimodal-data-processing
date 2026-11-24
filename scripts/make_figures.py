"""Generate every figure + metric used in the presentation.

    uv run python scripts/make_figures.py [names...]      # default: all figures

Figures -> reports/figures/*.png ; metrics -> reports/results_*.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lecturelens.speech.framing import window  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG, PROC, REP = ROOT / "reports/figures", ROOT / "data/processed", ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)
L1 = PROC / "lec01_what_is_computation"

# validated reference palette (dataviz skill): categorical slots 1-3, ink and surface
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
MOD_COLOR = {"speech": BLUE, "vision": ORANGE, "text": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "lines.linewidth": 2, "legend.frameon": False, "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def update_results(name: str, payload: dict):
    p = REP / f"results_{name}.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(payload)
    p.write_text(json.dumps(cur, indent=1, default=float))


def audio():
    from lecturelens.ingest import load_audio
    return load_audio(L1)


# ---- speech ------------------------------------------------------------------------------------------

def fig_windowing():
    x, sr = audio()
    from lecturelens.run import speech_tracks
    sp = speech_tracks(L1)
    i = int(np.flatnonzero((sp["t_pitch"] > 300) & (sp["f0"] > 0))[0])
    t0 = int(sp["t_pitch"][i] * sr)
    n = 512
    fr = x[t0:t0 + n]
    f = np.fft.rfftfreq(4096, 1 / sr)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(np.arange(n) / sr * 1000, fr, color=INK2, lw=1.2, label="frame")
    ax[0].plot(np.arange(n) / sr * 1000, window("hamming", n) * np.abs(fr).max(), color=BLUE, label="Hamming window")
    ax[0].set(xlabel="time (ms)", title="32 ms voiced frame")
    ax[0].legend(loc="lower right")
    for w, c, lab in [("rect", ORANGE, "Rectangular"), ("hamming", BLUE, "Hamming")]:
        S = 20 * np.log10(np.abs(np.fft.rfft(fr * window(w, n), 4096)) + 1e-9)
        ax[1].plot(f / 1000, S - S.max(), color=c, lw=1.4, label=lab)
    ax[1].set(xlim=(0, 4), ylim=(-80, 3), xlabel="frequency (kHz)", ylabel="dB", title="Spectral leakage")
    ax[1].legend()
    save(fig, "speech_windowing")


def fig_speech_tracks():
    from lecturelens.run import speech_tracks
    from lecturelens.speech.pitch import pitch_track
    from lecturelens.speech.spectral import spectrogram_db
    x, sr = audio()
    sp = speech_tracks(L1)
    a, b = 300.0, 320.0
    seg = x[int(a * sr):int(b * sr)]
    S, fq, tt = spectrogram_db(seg, sr)
    tc, f0c = pitch_track(seg, sr, method="cepstrum")
    sel = (sp["t"] >= a) & (sp["t"] < b)
    psel = (sp["t_pitch"] >= a) & (sp["t_pitch"] < b)
    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1.4, 1, 1]})
    ax[0].plot(np.arange(len(seg)) / sr + a, seg, color=INK2, lw=0.5)
    for s, e in sp["pauses"]:
        if e > a and s < b:
            for k in range(4):
                ax[k].axvspan(max(s, a), min(e, b), color=YELLOW, alpha=0.18, lw=0)
    ax[0].set(title="Waveform  (shaded = detected pauses)", yticks=[])
    ax[1].imshow(S, origin="lower", aspect="auto", extent=[a, b, 0, sr / 2000], cmap="Greys", vmin=S.max() - 70)
    ax[1].set(ylim=(0, 5), ylabel="kHz", title="Spectrogram (STFT, 32 ms Hann, 10 ms hop)")
    ax[1].grid(False)
    ax[2].plot(sp["t"][sel], sp["energy_db"][sel], color=BLUE, label="short-time energy (dB)")
    ax2 = ax[2].twinx()
    ax2.plot(sp["t"][sel], sp["zcr"][sel], color=ORANGE, lw=1, label="zero-crossing rate")
    ax2.set_yticks([])
    ax2.grid(False)
    ax[2].set(title="Short-time energy (blue) and zero-crossing rate (orange)")
    v = sp["f0"][psel] > 0
    ax[3].plot(sp["t_pitch"][psel][v], sp["f0"][psel][v], ".", ms=3, color=BLUE, label="autocorrelation")
    vc = f0c > 0
    ax[3].plot(tc[vc] + a, f0c[vc], ".", ms=3, color=ORANGE, label="cepstrum")
    ax[3].set(ylim=(60, 420), ylabel="F0 (Hz)", xlabel="lecture time (s)", title="Pitch")
    ax[3].legend(loc="upper right", markerscale=4, ncol=2)
    save(fig, "speech_tracks")
    # agreement of the two pitch trackers over the whole lecture
    tc_all, f0c_all = pitch_track(x[: int(600 * sr)], sr, method="cepstrum")
    fa = sp["f0"][: len(f0c_all)]
    both = (fa > 0) & (f0c_all > 0)
    rel = np.abs(fa[both] - f0c_all[both]) / fa[both]
    update_results("speech", {"pitch_agreement_within_5pct": float(np.mean(rel < 0.05)),
                              "pitch_octave_errors": float(np.mean(np.abs(np.log2(f0c_all[both] / fa[both])) > 0.8))})


def fig_mfcc():
    import librosa
    from lecturelens.speech.spectral import mfcc
    x, sr = audio()
    seg = x[int(300 * sr):int(305 * sr)]
    ours = mfcc(seg, sr)
    ref = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13, n_fft=512, hop_length=160, n_mels=40,
                               window="hann", center=True, pad_mode="constant")
    err = np.abs(ours - ref)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for a, M, t in [(ax[0], ours, "From scratch"), (ax[1], ref, "librosa")]:
        a.imshow(M, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-np.abs(ref).max() / 3, vmax=np.abs(ref).max() / 3)
        a.set(title=t, xlabel="frame (10 ms)")
        a.grid(False)
    ax[0].set_ylabel("coefficient")
    ax[2].imshow(err, origin="lower", aspect="auto", cmap="Greys")
    ax[2].set(title=f"|difference|  (max {err.max():.1e})", xlabel="frame (10 ms)")
    ax[2].grid(False)
    save(fig, "speech_mfcc_check")
    update_results("speech", {"mfcc_max_abs_diff_vs_librosa": float(err.max()),
                              "mfcc_rel_diff": float(err.max() / np.abs(ref).max())})


def fig_smart_speed():
    import soundfile as sf

    from lecturelens.run import speech_tracks
    from lecturelens.speech.ola import ola_reconstruct, remove_pauses, wsola
    x, sr = audio()
    sp = speech_tracks(L1)
    pauses = [tuple(p) for p in sp["pauses"]]
    trimmed, _ = remove_pauses(x, sr, pauses, keep_pause_s=0.15)
    fast = wsola(trimmed, 1.3, sr)
    naive_idx = np.arange(0, len(trimmed), 1.3).astype(int)
    durs = {"original": len(x) / sr / 60, "pauses trimmed": len(trimmed) / sr / 60, "+ 1.3x WSOLA": len(fast) / sr / 60}
    sf.write(L1 / "smart_speed.wav", fast.astype(np.float32), sr)
    seg = x[int(300 * sr):int(302 * sr)]
    recon = ola_reconstruct(seg)
    snr = 10 * np.log10(np.sum(seg ** 2) / np.sum((seg - recon) ** 2))

    from lecturelens.speech.pitch import pitch_track
    a = trimmed[int(200 * sr):int(215 * sr)]
    _, f_orig = pitch_track(a, sr)
    _, f_w = pitch_track(wsola(a, 1.3, sr), sr)
    _, f_n = pitch_track(a[np.arange(0, len(a), 1.3).astype(int)], sr)
    med = {k: float(np.median(v[v > 0])) for k, v in [("original", f_orig), ("wsola", f_w), ("naive", f_n)]}

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    names = list(durs)
    bars = ax[0].barh(names[::-1], [durs[n] for n in names[::-1]], color=[AQUA, BLUE, MUTED], height=0.55)
    for bar, n in zip(bars, names[::-1]):
        ax[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{durs[n]:.1f} min", va="center", color=INK)
    ax[0].set(xlabel="minutes", title="Lecture 1 listening time", xlim=(0, max(durs.values()) * 1.25))
    ax[0].grid(axis="y", visible=False)
    cats = ["original", "naive 1.3x\n(resample)", "WSOLA 1.3x"]
    vals = [med["original"], med["naive"], med["wsola"]]
    b2 = ax[1].bar(cats, vals, color=[MUTED, ORANGE, BLUE], width=0.55)
    for bb, v in zip(b2, vals):
        ax[1].text(bb.get_x() + bb.get_width() / 2, v + 4, f"{v:.0f} Hz", ha="center", color=INK)
    ax[1].set(ylabel="median F0 (Hz)", title="Speeding up without the chipmunk effect", ylim=(0, max(vals) * 1.2))
    ax[1].grid(axis="x", visible=False)
    save(fig, "speech_smart_speed")
    update_results("speech", {"smart_speed_minutes": durs, "ola_reconstruction_snr_db": float(snr),
                              "pitch_after_speedup_hz": med, "n_pauses_lec01": len(pauses)})


# ---- vision ------------------------------------------------------------------------------------------

def _frame(i, gray=False):
    return cv2.imread(str(sorted((L1 / "frames").glob("*.jpg"))[i]), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)


def fig_vision_enhance():
    from lecturelens.vision.enhance import auto_gamma, clahe, hist_equalize
    fs = pickle.load(open(L1 / "frames.pkl", "rb"))
    cam = int(np.flatnonzero(np.array(fs.types) == "camera")[200])
    g = _frame(cam, gray=True)
    outs = [("camera frame", g), ("auto-gamma", auto_gamma(g)), ("global hist. equalisation", hist_equalize(g)),
            ("CLAHE", clahe(g))]
    fig, ax = plt.subplots(2, 4, figsize=(16, 6), gridspec_kw={"height_ratios": [2.2, 1]})
    for k, (t, im) in enumerate(outs):
        ax[0, k].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[0, k].set_title(t)
        ax[0, k].axis("off")
        ax[1, k].bar(np.arange(256), np.bincount(im.ravel(), minlength=256), color=BLUE, width=1.0)
        ax[1, k].set(yticks=[], xlim=(0, 255))
        ax[1, k].grid(False)
    save(fig, "vision_enhancement")


def fig_frame_types():
    fs = pickle.load(open(L1 / "frames.pkl", "rb"))
    t = np.array(fs.types)
    picks = [("camera", int(np.flatnonzero(t == "camera")[300])), ("slide", int(np.flatnonzero(t == "slide")[150])),
             ("demo", int(np.flatnonzero(t == "demo")[40]))]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
    for a, (name, i) in zip(ax, picks):
        a.imshow(cv2.cvtColor(_frame(i), cv2.COLOR_BGR2RGB))
        a.set_title(f"{name}  ({(t == name).mean() * 100:.0f}% of frames)")
        a.axis("off")
    save(fig, "vision_frame_types")


def fig_restoration_and_ocr():
    from lecturelens.vision.enhance import blur, motion_psf, psnr, wiener_deconvolve
    from lecturelens.vision.lecture_vision import slide_crop
    from lecturelens.vision.ocr import char_accuracy, ocr, text_line_boxes
    segs = json.loads((L1 / "slides.json").read_text())
    good = [s for s in segs if s["type"] == "slide" and s.get("match_margin", 0) > 0.03 and len(s["pdf_text"]) > 60]
    # OCR accuracy: raw vs preprocessed against the exact PDF text
    rows = []
    for s in good:
        g = slide_crop(_frame(s["frame"], gray=True))
        rows.append({"raw": char_accuracy(ocr(g, preprocess=False), s["pdf_text"]),
                     "pre": char_accuracy(ocr(g, preprocess=True), s["pdf_text"])})
    acc = pd.DataFrame(rows)
    # restoration on a real slide crop with synthetic motion blur
    s = good[len(good) // 2]
    g = slide_crop(_frame(s["frame"], gray=True)).astype(float)
    psf = motion_psf(9, 0)
    rng = np.random.default_rng(0)
    bl = np.clip(blur(g, psf) + rng.normal(0, 1.0, g.shape), 0, 255).astype(np.uint8)
    rs = wiener_deconvolve(bl, psf, K=0.01)
    res = {"blur_psnr": psnr(bl, g), "wiener_psnr": psnr(rs, g),
           "blur_ocr_acc": char_accuracy(ocr(bl), s["pdf_text"]), "wiener_ocr_acc": char_accuracy(ocr(rs), s["pdf_text"]),
           "clean_ocr_acc": char_accuracy(ocr(g.astype(np.uint8)), s["pdf_text"])}
    boxes, bw = text_line_boxes(g.astype(np.uint8))
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    vis = cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for (x0, y0, w, h) in boxes:
        cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (42, 120, 214), 2)
    for a, im, t in [(ax[0], bw, "Otsu + opening (text = white)"), (ax[1], vis, f"dilation -> {len(boxes)} text lines"),
                     (ax[2], bl, f"motion blur  PSNR {res['blur_psnr']:.1f} dB, OCR {res['blur_ocr_acc'] * 100:.0f}%"),
                     (ax[3], rs, f"Wiener  PSNR {res['wiener_psnr']:.1f} dB, OCR {res['wiener_ocr_acc'] * 100:.0f}%")]:
        a.imshow(im, cmap="gray")
        a.set_title(t, fontsize=11)
        a.axis("off")
    save(fig, "vision_ocr_restoration")
    update_results("vision", {"ocr": {"n_slides": len(acc), "acc_raw_mean": acc.raw.mean(), "acc_pre_mean": acc.pre.mean(),
                                      "acc_raw_median": acc.raw.median(), "acc_pre_median": acc.pre.median()},
                              "restoration": res})


def fig_change_detection():
    from lecturelens.vision.slides import change_prf, detect_changes, robust_z, shot_aware_scores
    fs = pickle.load(open(L1 / "frames.pkl", "rb"))
    gt = pd.read_csv(ROOT / "annotations/lec01_visual_events.csv")
    res = json.loads((REP / "results_vision.json").read_text())["visual_events_lec01"]["results"]
    z = shot_aware_scores(fs.dists, fs.types)
    pred = detect_changes(z, fs.times, k=1.0, already_z=True, min_gap_s=3.0)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.42})
    a, b = 480, 780
    m = (fs.times >= a) & (fs.times < b)
    ax[0].plot(fs.times[m], np.clip(z[m], 0, 6), color=BLUE, lw=1.2, label="shot-aware score")
    ax[0].axhline(1, color=INK2, lw=1, ls="--")
    for t in gt.time_s[(gt.time_s >= a) & (gt.time_s < b)]:
        ax[0].axvline(t, color=ORANGE, lw=1, alpha=0.6)
    pr = [p for p in pred if a <= p < b]
    ax[0].plot(pr, [6.4] * len(pr), "v", color=BLUE, ms=8, label="detected")
    ax[0].plot([], [], color=ORANGE, label="ground truth")
    ax[0].set(xlabel="lecture time (s)", ylabel="score (clipped)", ylim=(-0.2, 7),
              title="Change detection, minutes 8–13 of Lecture 1")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    names = list(res)
    f1 = [res[n]["F1"] for n in names]
    build = [res[n]["recall_slide_build"] for n in names]
    y = np.arange(len(names))
    ax[1].barh(y - 0.19, f1, height=0.36, color=BLUE, label="F1, all 139 events")
    ax[1].barh(y + 0.19, build, height=0.36, color=ORANGE, label="recall, 6 slide builds")
    for yy, v, w in zip(y, f1, build):
        ax[1].text(v + 0.015, yy - 0.19, f"{v:.2f}", va="center", fontsize=10, color=INK)
        ax[1].text(w + 0.015, yy + 0.19, f"{w:.2f}", va="center", fontsize=10, color=INK)
    ax[1].set(yticks=y, yticklabels=names, xlim=(0, 1.15), title="Descriptor comparison")
    ax[1].invert_yaxis()
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2, fontsize=10)
    ax[1].grid(axis="y", visible=False)
    save(fig, "vision_change_detection")


# ---- text --------------------------------------------------------------------------------------------

def _all_words():
    from lecturelens.text.asr import words_of
    out = {}
    for d in sorted(PROC.iterdir()):
        if (d / "transcript.json").exists():
            out[d.name] = words_of(json.loads((d / "transcript.json").read_text()))
    return out


def fig_zipf_heaps():
    from lecturelens.text.normalize import tokenize
    from lecturelens.text.stats import heaps, type_token_ratio, zipf
    from lecturelens.vision.deck import load_deck
    spoken = [t for ws in _all_words().values() for t in tokenize(" ".join(w["word"] for w in ws))]
    slides = [t for n in [1, 8, 10] for p in load_deck(ROOT / f"data/slides/lec{n}.pdf") for t in tokenize(p.text)]
    # compare on equal-length samples for Heaps/TTR (both are length dependent)
    n = min(len(spoken), len(slides))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    out = {}
    for toks, c, lab in [(spoken, BLUE, "spoken (transcripts)"), (slides, ORANGE, "written (slides)")]:
        r, f, s, C = zipf(toks)
        ax[0].loglog(r, f, ".", ms=3, color=c, label=f"{lab}: s = {s:.2f}")
        ax[0].loglog(r, C * r ** -s, "-", lw=1, color=c, alpha=0.6)
        nn, V, K, beta = heaps(toks[:n])
        ax[1].plot(nn, V, color=c, label=f"{lab}: β = {beta:.2f}")
        out[lab] = {"tokens": len(toks), "zipf_s": s, "heaps_K": K, "heaps_beta": beta,
                    "ttr_equal_len": type_token_ratio(toks[:n]), "vocab": len(set(toks))}
    ax[0].set(xlabel="rank", ylabel="frequency", title="Zipf's law")
    ax[0].legend()
    ax[1].set(xlabel="tokens seen", ylabel="vocabulary size", title=f"Heaps' law (first {n:,} tokens)")
    ax[1].legend()
    save(fig, "text_zipf_heaps")
    update_results("text", {"zipf_heaps": out})


def text_bpe_lm():
    import tiktoken

    from lecturelens.text.bpe import BPE
    from lecturelens.text.lm import BigramLM
    from lecturelens.text.normalize import lemma, porter_stem, tokenize
    words = _all_words()
    names = list(words)
    corpus = [t for ws in words.values() for t in tokenize(" ".join(w["word"] for w in ws))]
    bpe = BPE.train(corpus, num_merges=800)
    enc = tiktoken.get_encoding("cl100k_base")
    probe = ["computation", "imperative", "declarative", "polymorphism", "inheritance", "iteratively",
             "algorithm", "efficiently", "getattr", "isinstance"]
    splits = {w: {"bpe_course": bpe.encode_word(w), "gpt_cl100k": [enc.decode([t]) for t in enc.encode(" " + w)]}
              for w in probe}
    morph = {w: {"porter": porter_stem(w), "lemma": lemma(w)} for w in
             ["computations", "computing", "programs", "studies", "better", "was", "classes", "running"]}
    # LM smoothing comparison: train on 2 lectures, test on the third
    sents = {n: [tokenize(s["text"]) for s in json.loads((PROC / n / "transcript.json").read_text())] for n in names}
    ppl = {}
    for held in names:
        train = [s for n in names if n != held for s in sents[n]]
        ppl[held] = {sm: BigramLM(train, smoothing=sm, k=0.05, min_count=2).perplexity(sents[held])
                     for sm in ["laplace", "addk", "kn"]}
    update_results("text", {"bpe_splits": splits, "morphology": morph, "lm_perplexity": ppl,
                            "bpe_merges": len(bpe.merges)})
    print(json.dumps(ppl, indent=1))


def text_search_eval():
    """Known-item search: query = slide title from the PDF, relevant = when that slide is on screen."""
    from lecturelens.text.asr import words_of
    from lecturelens.text.lecture_text import LectureSearch, make_chunks
    from lecturelens.vision.deck import load_deck
    rows = []
    for d in sorted(PROC.iterdir()):
        if not (d / "slides.json").exists():
            continue
        segs = json.loads((d / "slides.json").read_text())
        words = words_of(json.loads((d / "transcript.json").read_text()))
        meta = json.loads((d / "meta.json").read_text())
        n = int(d.name[3:5])
        deck = load_deck(ROOT / f"data/slides/lec{n}.pdf")
        # relevant intervals per page from the slide segments + the following 60 s (talk continues on camera)
        intervals: dict[int, list] = {}
        for s in segs:
            if s["type"] == "slide" and s["page"] is not None and s.get("match_margin", 0) > 0.03:
                intervals.setdefault(s["page"], []).append((s["start"], s["end"] + 60))
        for use_slides in [False, True]:
            for method in ["bm25", "tfidf"]:
                chunks = make_chunks(words, [] if not use_slides else
                                     [dict(s, ocr=s["ocr"]) for s in segs if s["type"] == "slide"],
                                     duration=meta["duration"])
                idx = LectureSearch(chunks, use_slides=use_slides, method=method)
                for pg, iv in intervals.items():
                    title = " ".join(deck[pg].text.split()[:6])
                    hits = idx.search(title, k=10)
                    rr = 0.0
                    for rank, (_, ch) in enumerate(hits, 1):
                        if any(ch.start < e and ch.end > s for s, e in iv):
                            rr = 1 / rank
                            break
                    rows.append({"lecture": d.name, "method": method, "slides_indexed": use_slides,
                                 "query": title, "rr": rr, "hit@1": rr == 1.0, "hit@5": rr >= 0.2})
    df = pd.DataFrame(rows)
    summ = df.groupby(["method", "slides_indexed"]).agg(MRR=("rr", "mean"), hit1=("hit@1", "mean"),
                                                        hit5=("hit@5", "mean"), n=("rr", "size")).reset_index()
    print(summ.round(3).to_string())
    update_results("text", {"search": summ.to_dict(orient="records")})
    return summ


# ---- biomedical --------------------------------------------------------------------------------------

def fig_rppg_synthetic():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_bio import synthetic_face_rgb

    from lecturelens.bio.rppg import bandpass, green, pos
    fs = 30.0
    rgb, t = synthetic_face_rgb(100, fs, 30)
    sigs = {"green channel": bandpass(green(rgb), fs), "POS": bandpass(pos(rgb, fs), fs)}
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for (lab, s), c in zip(sigs.items(), [AQUA, BLUE]):
        ax[0].plot(t[:300], s[:300] / np.abs(s[:300]).max(), color=c, lw=1.4, label=lab)
        n = len(s) * 8
        spec = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
        f = np.fft.rfftfreq(n, 1 / fs) * 60
        ax[1].plot(f, spec / spec.max(), color=c, label=lab)
    ax[1].axvline(100, color=INK2, ls="--", lw=1)
    ax[1].text(101, 0.95, "true HR 100 bpm", color=INK2, fontsize=10)
    ax[1].axvline(78, color=ORANGE, ls=":", lw=1.5)
    ax[1].text(52, 0.75, "lighting\nflicker", color=ORANGE, fontsize=10)
    ax[0].set(xlabel="s", title="Recovered pulse (first 10 s)", yticks=[])
    ax[0].legend(loc="lower right", ncol=2)
    ax[1].set(xlim=(40, 180), xlabel="beats per minute", title="Spectrum: POS rejects the flicker", yticks=[])
    ax[1].legend(loc="center right")
    save(fig, "bio_rppg_synthetic")


# ---- overview ----------------------------------------------------------------------------------------

def fig_timeline():
    df = pd.read_parquet(L1 / "timeline.parquet")
    segs = json.loads((L1 / "slides.json").read_text())
    tc = (df.t0 + df.t1) / 2 / 60
    fig, ax = plt.subplots(4, 1, figsize=(15, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1, 0.6]})
    col = "importance_learned" if "importance_learned" in df else "importance"
    lab = pd.read_csv(ROOT / "annotations/labels_lec01_what_is_computation.csv").label.to_numpy()[: len(df)]
    ax[0].fill_between(tc, df[col], color=VIOLET, alpha=0.25, lw=0)
    ax[0].plot(tc, df[col], color=VIOLET, lw=1.4, label="model")
    ax[0].plot(tc[lab == 1], np.full((lab == 1).sum(), df[col].max() * 1.08), "|", color=ORANGE, ms=9, mew=2,
               label="human key moment")
    ax[0].legend(loc="upper right", ncol=2, fontsize=10)
    ax[0].set(title="Key-moment probability (early fusion, trained on the other two lectures)", yticks=[])
    ax[1].plot(tc, df.energy_db, color=BLUE, lw=1.2)
    ax[1].set(title="Speech: energy (dB)")
    ax[2].plot(tc, df.salience, color=AQUA, lw=1.2)
    ax[2].set(title="Text: salience (mean IDF of spoken content words)")
    colors = {"slide": BLUE, "camera": MUTED, "demo": ORANGE}
    for s in segs:
        ax[3].barh(0, (s["end"] - s["start"]) / 60, left=s["start"] / 60, color=colors[s["type"]], height=0.8, lw=0)
    for k, c in colors.items():
        ax[3].barh(0, 0, color=c, label=k)
    ax[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=3)
    ax[3].set(yticks=[], title="Vision: what is on screen", xlabel="lecture time (min)")
    ax[3].grid(False)
    save(fig, "overview_timeline")


def fig_fusion():
    p = REP / "results_fusion.json"
    if not p.exists():
        print("skip fusion (no results)")
        return
    res = pd.DataFrame(json.loads(p.read_text())["loo"])
    order = ["speech", "vision", "text", "early", "late", "unsupervised"]
    g = res.groupby("model")[["AP", "F1"]].mean().reindex(order)
    base = res.base_rate.mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [BLUE, ORANGE, AQUA, VIOLET, VIOLET, MUTED]
    bars = ax.bar(order, g.AP, color=colors, width=0.6)
    bars[4].set_hatch("//")
    bars[4].set_edgecolor(SURF)
    for b, v in zip(bars, g.AP):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", color=INK)
    ax.axhline(base, color=INK2, ls="--", lw=1)
    ax.text(5.35, base + 0.01, f"chance = {base:.2f}", color=INK2, ha="right", fontsize=10)
    ax.set(ylabel="average precision", title="Key-moment detection, leave-one-lecture-out (mean of 3 folds)",
           ylim=(0, max(g.AP) * 1.25))
    ax.grid(axis="x", visible=False)
    save(fig, "fusion_ablation")


ALL = {f.__name__: f for f in [fig_windowing, fig_speech_tracks, fig_mfcc, fig_smart_speed, fig_frame_types,
                               fig_vision_enhance, fig_restoration_and_ocr, fig_change_detection, fig_zipf_heaps,
                               text_bpe_lm, text_search_eval, fig_rppg_synthetic, fig_timeline, fig_fusion]}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
