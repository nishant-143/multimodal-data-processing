"""Key-moment detection: unimodal vs early vs late fusion, leave-one-lecture-out.

Two feature sets are compared:
  * base      - the signal-level features (prosody, visual activity, salience/novelty/perplexity)
  * enriched  - base + discourse cues (text) + temporal context (±1-window mean of every feature)

    uv run python scripts/eval_fusion.py   ->  reports/results_fusion.json
"""
import json
from pathlib import Path

import pandas as pd

from lecturelens.fusion.importance import MODALITIES, coefficients, evaluate

PROC, ANN = Path("data/processed"), Path("annotations")
CUES = ["cue_define", "cue_rule", "cue_flag"]

base = {m: list(c) for m, c in MODALITIES.items()}
enriched = {m: list(c) + [f"{x}_ctx" for x in c] for m, c in base.items()}
enriched["text"] += CUES + [f"{x}_ctx" for x in CUES]
all_feats = sorted({c for cols in enriched.values() for c in cols if not c.endswith("_ctx")})

frames = []
for d in sorted(PROC.iterdir()):
    lab_p = ANN / f"labels_{d.name}.csv"
    if not lab_p.exists():
        continue
    df = pd.read_parquet(d / "timeline.parquet")
    lab = pd.read_csv(lab_p)
    df["label"] = lab.set_index("window").label.reindex(range(len(df))).fillna(0).astype(int).to_numpy()
    # temporal context: centred 3-window mean (a 10 s window often splits the key sentence)
    for c in all_feats:
        df[f"{c}_ctx"] = df[c].rolling(3, center=True, min_periods=1).mean()
    df = df[~df.boilerplate & (df.speech_ratio > 0.05)]          # cleaned windows only
    # per-lecture standardisation: speakers differ (F0 ~220 Hz vs ~136 Hz), rooms differ
    cols = all_feats + [f"{c}_ctx" for c in all_feats]
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0).replace(0, 1)
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

out = {"n_windows": int(len(data)), "n_pos": int(data.label.sum()), "base_rate": float(data.label.mean())}
for name, mods in [("base", base), ("enriched", enriched)]:
    res = evaluate(data, modalities=mods)
    summary = res.groupby("model")[["AP", "F1"]].mean()
    print(f"\n== {name} ==\n", summary.round(3).to_string())
    drop = {}
    for m in mods:
        r = evaluate(data, modalities={k: v for k, v in mods.items() if k != m})
        drop[m] = float(r[r.model == "early"].AP.mean())
    print("early-fusion AP without:", {k: round(v, 3) for k, v in drop.items()})
    out[name] = {"loo": res.to_dict(orient="records"), "summary": summary.reset_index().to_dict(orient="records"),
                 "early_AP_without": drop}
coef = coefficients(data, cols=[c for cols in enriched.values() for c in cols if not c.endswith("_ctx")])
print("\ncoefficients (enriched, no context):\n", coef.round(2).to_string())
out["coefficients"] = coef.to_dict()
out["loo"] = out["enriched"]["loo"]          # used by the figure
Path("reports/results_fusion.json").write_text(json.dumps(out, indent=1, default=float))

# ---- learned importance for the dashboard: early fusion (enriched) trained on the OTHER lectures -----
from lecturelens.fusion.importance import _model  # noqa: E402

cols = [c for v in enriched.values() for c in v]
for held in data.lecture.unique():
    tr = data[data.lecture != held]
    m = _model().fit(tr[cols].fillna(0), tr.label)
    tl_p = PROC / held / "timeline.parquet"
    tl = pd.read_parquet(tl_p)
    te = data[data.lecture == held]
    prob = pd.Series(m.predict_proba(te[cols].fillna(0))[:, 1], index=te.t0.to_numpy())
    tl["importance_learned"] = tl.t0.map(prob).fillna(0.0).to_numpy()
    tl.to_parquet(tl_p)
    top = tl.nlargest(4, "importance_learned")
    print(held[:5], "learned top:", [f"{int(t // 60)}:{int(t % 60):02d} {x[:50]}" for t, x in zip(top.t0, top.text)])
