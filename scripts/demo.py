"""Terminal demo of LectureLens on a processed lecture.

    uv run python scripts/demo.py                                  # lecture 1, default queries
    uv run python scripts/demo.py --lecture lec10 --query "big O notation"
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from lecturelens.text.asr import words_of
from lecturelens.text.lecture_text import LectureSearch, make_chunks

PROC = Path("data/processed")


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


ap = argparse.ArgumentParser()
ap.add_argument("--lecture", default="lec01")
ap.add_argument("--query", action="append")
a = ap.parse_args()
d = next(p for p in sorted(PROC.iterdir()) if p.name.startswith(a.lecture))
df = pd.read_parquet(d / "timeline.parquet")
segs = json.loads((d / "slides.json").read_text())
words = words_of(json.loads((d / "transcript.json").read_text()))
meta = json.loads((d / "meta.json").read_text())
sp = json.loads(Path("reports/results_speech.json").read_text())

print(f"\n=== {d.name}  ({meta['duration'] / 60:.1f} min) ===")
types = pd.Series([s["type"] for s in segs]).value_counts().to_dict()
print(f"visual segments: {len(segs)}  {types}   |   spoken words: {len(words):,}")

print("\n-- Ask the lecture (BM25 over transcript + slide text) --")
idx = LectureSearch(make_chunks(words, [dict(s, ocr=s["pdf_text"] or s["ocr"]) for s in segs],
                                duration=meta["duration"]))
for q in a.query or ["what is an algorithm", "difference between declarative and imperative knowledge"]:
    print(f'\nQ: "{q}"')
    for score, ch in idx.search(q, k=3):
        print(f"  {mmss(ch.start):>6}  score {score:5.2f}  {ch.text[:95]}…")

col = "importance_learned" if "importance_learned" in df else "importance"
print(f"\n-- Highlights ({'learned early-fusion model' if col.endswith('learned') else 'hand-weighted'}) --")
for _, r in df.nlargest(6, col).sort_values("t0").iterrows():
    print(f"  {mmss(r.t0):>6}  {r.text[:100]}…")

ins_p = d / "insights.json"
if ins_p.exists():
    print("\n-- Off-slide insights (said, but not on the slide) --")
    for x in json.loads(ins_p.read_text())[:3]:
        print(f"  {mmss(x['start']):>6}  {x['text'][:100]}…")

ss = d / "smart_speed.wav"
if ss.exists():
    import soundfile as sf
    print(f"\n-- Smart-speed audio: {meta['duration'] / 60:.1f} min -> {sf.info(str(ss)).duration / 60:.1f} min  ({ss})")
print()
