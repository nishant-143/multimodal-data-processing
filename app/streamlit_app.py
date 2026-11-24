"""LectureLens dashboard:  uv run streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lecturelens.text.asr import words_of
from lecturelens.text.lecture_text import LectureSearch, make_chunks

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
st.set_page_config(page_title="LectureLens", layout="wide")


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


@st.cache_data
def load(name: str, mtime: float = 0.0):            # mtime in the key: re-read when the pipeline reruns
    d = PROC / name
    df = pd.read_parquet(d / "timeline.parquet")
    segs = json.loads((d / "slides.json").read_text())
    words = words_of(json.loads((d / "transcript.json").read_text()))
    ins = json.loads((d / "insights.json").read_text()) if (d / "insights.json").exists() else []
    sp = dict(np.load(d / "speech.npz", allow_pickle=True))
    return df, segs, words, ins, sp


@st.cache_resource
def search_index(name: str, mtime: float = 0.0):
    df, segs, words, _, _ = load(name, mtime)
    slide_segs = [dict(s, ocr=(s["pdf_text"] or s["ocr"])) for s in segs]
    return LectureSearch(make_chunks(words, slide_segs, duration=float(df["t1"].max())))


lectures = sorted(p.name for p in PROC.iterdir() if (p / "timeline.parquet").exists())
if not lectures:
    st.error("No processed lectures. Run `uv run python -m lecturelens.run` first.")
    st.stop()

st.sidebar.title("LectureLens")
name = st.sidebar.selectbox("Lecture", lectures, format_func=lambda s: s.replace("_", " "))
_mt = (PROC / name / "timeline.parquet").stat().st_mtime
df, segs, words, insights, sp = load(name, _mt)
if "t" not in st.session_state or st.session_state.get("lec") != name:
    st.session_state.update(t=0, lec=name)

mode = st.sidebar.radio("Highlight model", ["learned (early fusion)", "hand-weighted"],
                        help="learned: logistic regression trained on the *other* lectures' key-moment labels")
w = {m: st.sidebar.slider(f"{m} weight", 0.0, 2.0, 1.0, 0.1, disabled=mode.startswith("learned"))
     for m in ["speech", "vision", "text"]}
from lecturelens.fusion.importance import unsupervised_score  # noqa: E402

feat_cols = [c for c in df.columns
             if c not in ("lecture", "t0", "t1", "text", "slide_text", "importance", "label", "boilerplate",
                          "importance_learned")]
_imp = unsupervised_score(df.loc[~df.boilerplate, feat_cols], w)
df["importance"] = _imp.reindex(df.index).fillna(_imp.min())
if mode.startswith("learned") and "importance_learned" in df:
    df["importance"] = df["importance_learned"]

# ---- search ---------------------------------------------------------------------------------------
q = st.text_input("Ask the lecture (BM25 over transcript + slide text)", placeholder="e.g. what is an algorithm")
if q:
    hits = search_index(name, _mt).search(q, k=5)
    cols = st.columns(max(len(hits), 1))
    for c, (score, ch) in zip(cols, hits):
        with c:
            if st.button(f"{mmss(ch.start)}  (score {score:.1f})", key=f"hit{ch.start}"):
                st.session_state.t = int(ch.start)
            st.caption(ch.text[:160] + "…")

left, right = st.columns([3, 2])
with left:
    st.video(str(PROC / name / "proxy.mp4"), start_time=int(st.session_state.t))
with right:
    st.subheader("Highlights")
    top = df.nlargest(8, "importance").sort_values("t0")
    for _, r in top.iterrows():
        if st.button(f"▶ {mmss(r.t0)} — {r.text[:70]}…", key=f"hl{r.t0}"):
            st.session_state.t = int(r.t0)

# ---- synchronised timeline ---------------------------------------------------------------------------
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.26, 0.16, 0.16, 0.16, 0.26],
                    subplot_titles=["Importance (fused)", "Energy (dB)", "Pitch (Hz)", "Visual activity", "Slides"])
tc = (df.t0 + df.t1) / 2
fig.add_trace(go.Scatter(x=tc, y=df.importance, fill="tozeroy", name="importance", line=dict(color="#6250d6")), 1, 1)
fig.add_trace(go.Scatter(x=tc, y=df.energy_db, name="energy", line=dict(color="#2a78d6")), 2, 1)
fig.add_trace(go.Scatter(x=tc, y=df.pitch_mean, name="pitch", line=dict(color="#2a78d6")), 3, 1)
fig.add_trace(go.Scatter(x=tc, y=df.diff_energy, name="frame diff", line=dict(color="#eb6834")), 4, 1)
colors = {"slide": "#2a78d6", "camera": "#9d9d9d", "demo": "#eb6834"}
for kind, col in colors.items():          # one trace per type, so every bar gets the full row height
    ss = [s for s in segs if s["type"] == kind]
    if ss:
        fig.add_trace(go.Bar(x=[s["end"] - s["start"] for s in ss], base=[s["start"] for s in ss],
                             y=[kind] * len(ss), orientation="h", marker_color=col, name=kind, width=0.8,
                             hovertext=[(s["pdf_text"] or s["ocr"])[:80] for s in ss]), 5, 1)
fig.update_layout(barmode="overlay")
fig.add_vline(x=st.session_state.t, line_color="red")
fig.update_layout(height=620, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---- off-slide insights --------------------------------------------------------------------------------
st.subheader("Off-slide insights — said, but not on the slide")
for d in insights[:8]:
    c1, c2 = st.columns([1, 9])
    if c1.button(mmss(d["start"]), key=f"ins{d['start']}"):
        st.session_state.t = int(d["start"])
    c2.write(f"{d['text']}  \n*slide similarity {d['slide_cos']:.2f} · salience {d['salience']:.2f}*")

ss = PROC / name / "smart_speed.wav"
if ss.exists():
    st.subheader("Smart-speed audio (pauses trimmed + 1.3× WSOLA)")
    st.audio(str(ss))
