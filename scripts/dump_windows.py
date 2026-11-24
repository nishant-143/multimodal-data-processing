"""Write 10 s transcript windows to annotations/windows_<lecture>.tsv for key-moment labelling."""
import json
from pathlib import Path

from lecturelens.fusion.align import text_in_windows, window_edges
from lecturelens.text.asr import words_of

PROC = Path("data/processed")
for d in sorted(PROC.iterdir()):
    if not (d / "transcript.json").exists():
        continue
    meta = json.loads((d / "meta.json").read_text())
    words = words_of(json.loads((d / "transcript.json").read_text()))
    E = window_edges(meta["duration"], 10.0)
    texts = text_in_windows(words, E)
    out = Path("annotations") / f"windows_{d.name}.tsv"
    with out.open("w") as f:
        f.write("window\tstart\ttext\n")
        for i, t in enumerate(texts):
            f.write(f"{i}\t{int(E[i]) // 60}:{int(E[i]) % 60:02d}\t{t}\n")
    print(out, len(texts))
