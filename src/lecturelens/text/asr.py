"""Speech-to-text with word timestamps (faster-whisper, CPU int8). The only neural component."""
from __future__ import annotations

import json
from pathlib import Path


def transcribe(audio: Path, out_json: Path, model_size: str = "small.en", force: bool = False) -> list[dict]:
    """Returns [{start, end, text, words:[{start,end,word,prob}]}] and caches to out_json."""
    if out_json.exists() and not force:
        return json.loads(out_json.read_text())
    import soundfile as sf
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # pass samples directly (16 kHz mono from ingest) — avoids PyAV version issues in decode_audio
    x, sr = sf.read(str(audio), dtype="float32")
    assert sr == 16000, "ingest() writes 16 kHz audio"
    segments, _ = model.transcribe(x, word_timestamps=True, vad_filter=False, beam_size=5,
                                   condition_on_previous_text=False)
    out = []
    for s in segments:
        out.append({
            "start": s.start, "end": s.end, "text": s.text.strip(),
            "words": [{"start": w.start, "end": w.end, "word": w.word.strip(), "prob": w.probability}
                      for w in (s.words or [])],
        })
    out_json.write_text(json.dumps(out, indent=1))
    return out


def words_of(segments: list[dict]) -> list[dict]:
    return [w for s in segments for w in s["words"]]
