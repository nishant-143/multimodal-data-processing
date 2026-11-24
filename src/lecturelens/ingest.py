"""Data collection & cleaning: split a lecture video into 16 kHz mono audio + 1 fps grey frames."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf


@dataclass
class Meta:
    video: str
    duration: float
    fps: float
    width: int
    height: int
    audio_sr: int
    frame_rate: float


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                          str(video)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def ingest(video: Path, out_dir: Path, sr: int = 16000, frame_rate: float = 1.0, width: int = 1280,
           start: float | None = None, duration: float | None = None, force: bool = False) -> Meta:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_p = out_dir / "meta.json"
    if meta_p.exists() and not force:
        return Meta(**json.loads(meta_p.read_text()))
    trim = (["-ss", str(start)] if start else []) + (["-t", str(duration)] if duration else [])

    # audio: mono, resampled, loudness-normalised (EBU R128) to make energy thresholds comparable
    subprocess.run(["ffmpeg", "-y", "-v", "error", *trim, "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr),
                    "-af", "loudnorm=I=-23:TP=-2", str(out_dir / "audio.wav")], check=True)
    # frames: 1 fps, scaled, greyscale + a colour copy for the dashboard
    fdir = out_dir / "frames"
    fdir.mkdir(exist_ok=True)
    for f in fdir.glob("*.jpg"):
        f.unlink()
    subprocess.run(["ffmpeg", "-y", "-v", "error", *trim, "-i", str(video), "-vf",
                    f"fps={frame_rate},scale={width}:-2", "-q:v", "3", str(fdir / "%05d.jpg")], check=True)
    # a small playable proxy for the dashboard
    subprocess.run(["ffmpeg", "-y", "-v", "error", *trim, "-i", str(video), "-vf", "scale=854:-2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-c:a", "aac", "-b:a", "96k",
                    str(out_dir / "proxy.mp4")], check=True)

    info = probe(video)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = map(float, vs.get("avg_frame_rate", "30/1").split("/"))
    dur = sf.info(str(out_dir / "audio.wav")).duration
    meta = Meta(str(video), dur, num / den if den else 30.0, int(vs["width"]), int(vs["height"]), sr, frame_rate)
    meta_p.write_text(json.dumps(asdict(meta), indent=2))
    return meta


def load_audio(out_dir: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(out_dir / "audio.wav"), dtype="float32")
    return x, sr


def load_frames(out_dir: Path, gray: bool = True) -> tuple[list[np.ndarray], np.ndarray]:
    files = sorted((out_dir / "frames").glob("*.jpg"))
    meta = json.loads((out_dir / "meta.json").read_text())
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    frames = [cv2.imread(str(f), flag) for f in files]
    times = (np.arange(len(files)) + 0.5) / meta["frame_rate"]   # ffmpeg fps filter samples mid-interval
    return frames, times
