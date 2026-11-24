"""Learner-pulse study: record your webcam while you watch a lecture, then analyse it.

1) Record (run it yourself; nothing is uploaded anywhere):
       uv run python scripts/learner_session.py record --lecture lec01 --minutes 10
   - Start the lecture video at 0:00 (dashboard or any player) when the countdown hits zero.
   - Sit still-ish, face the camera, normal room light. Note your smartwatch / phone HR every ~2 min
     into annotations/learner_<lecture>_reference_hr.csv  (columns: lecture_time_s,bpm).
2) Analyse:
       uv run python scripts/learner_session.py analyse --lecture lec01
   -> data/processed/<lecture>/learner.json  (HR track, blinks, rPPG MAE vs reference, correlation
      of HR / blink rate with the lecture importance curve) and reports/figures/bio_learner.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lecturelens.bio.rppg import bandpass, blinks, extract_face_signals, green, hr_track, pos, rmssd_ms

PROC = Path("data/processed")


def lecture_dir(prefix: str) -> Path:
    return next(d for d in sorted(PROC.iterdir()) if d.name.startswith(prefix))


def record(lecture: str, minutes: float, cam: int = 0):
    d = lecture_dir(lecture)
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (macOS: allow camera access for your terminal).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for s in (3, 2, 1):
        print(f"start the lecture at 0:00 in {s}…", flush=True)
        time.sleep(1)
    print("GO — recording", flush=True)
    out_p = d / "learner.mp4"
    w = cv2.VideoWriter(str(out_p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    t0, n = time.time(), 0
    while time.time() - t0 < minutes * 60:
        ok, f = cap.read()
        if not ok:
            break
        w.write(cv2.resize(f, (640, 480)))
        n += 1
    w.release()
    cap.release()
    fps = n / (time.time() - t0)
    (d / "learner_meta.json").write_text(json.dumps({"fps_measured": fps, "frames": n, "lecture_offset_s": 0.0}))
    print(f"saved {out_p} ({n} frames, {fps:.1f} fps)")


def analyse(lecture: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = lecture_dir(lecture)
    meta = json.loads((d / "learner_meta.json").read_text())
    sig = extract_face_signals(d / "learner.mp4")
    fs = meta.get("fps_measured", sig.fs)          # webcams rarely deliver the nominal 30 fps
    off = meta.get("lecture_offset_s", 0.0)
    out = {}
    for name, raw in [("pos", pos(sig.rgb, fs)), ("green", green(sig.rgb))]:
        p = bandpass(raw, fs)
        t, hr = hr_track(p, fs)
        out[name] = {"t": (t + off).tolist(), "hr": hr.tolist(), "rmssd_ms": rmssd_ms(p, fs)}
    bl = [b + off for b in blinks(sig.ear, fs)]
    out["blinks"] = bl
    ref_p = Path("annotations") / f"learner_{d.name[:5]}_reference_hr.csv"
    if ref_p.exists():
        ref = pd.read_csv(ref_p)
        for name in ["pos", "green"]:
            est = np.interp(ref.lecture_time_s, out[name]["t"], out[name]["hr"])
            out[name]["mae_bpm_vs_reference"] = float(np.mean(np.abs(est - ref.bpm)))
    # relate to the lecture importance curve (10 s windows)
    tl = pd.read_parquet(d / "timeline.parquet")
    tc = (tl.t0 + tl.t1) / 2
    hr_w = np.interp(tc, out["pos"]["t"], out["pos"]["hr"], left=np.nan, right=np.nan)
    blink_w = np.histogram(bl, bins=np.r_[tl.t0, tl.t1.iloc[-1]])[0] * 6.0   # per minute
    m = ~np.isnan(hr_w)
    out["corr_importance_hr"] = float(np.corrcoef(tl.importance[m], hr_w[m])[0, 1])
    out["corr_importance_blink_rate"] = float(np.corrcoef(tl.importance[m], blink_w[m])[0, 1])
    (d / "learner.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(tc[m] / 60, tl.importance[m], color="#6250d6")
    ax[0].set_title("Lecture importance")
    ax[1].plot(np.array(out["pos"]["t"]) / 60, out["pos"]["hr"], color="#2a78d6", label="rPPG (POS)")
    if ref_p.exists():
        ax[1].plot(ref.lecture_time_s / 60, ref.bpm, "o", color="#eb6834", label="smartwatch")
    ax[1].set(title="Learner heart rate (bpm)")
    ax[1].legend()
    ax[2].bar(tc[m] / 60, blink_w[m], width=0.15, color="#1baf7a")
    ax[2].set(title="Blinks per minute", xlabel="lecture time (min)")
    fig.savefig("reports/figures/bio_learner.png", dpi=200, bbox_inches="tight")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))}, indent=1))
    print({k: {kk: vv for kk, vv in out[k].items() if not isinstance(vv, list)} for k in ["pos", "green"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "analyse"])
    ap.add_argument("--lecture", default="lec01")
    ap.add_argument("--minutes", type=float, default=10)
    a = ap.parse_args()
    record(a.lecture, a.minutes) if a.cmd == "record" else analyse(a.lecture)
