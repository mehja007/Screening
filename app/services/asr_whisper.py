from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from faster_whisper import WhisperModel


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_ffmpeg_to_wav(input_path: Path, output_wav: Path) -> None:
    """
    Convert input audio to WAV mono 16kHz (best for ASR).
    Requires ffmpeg available in PATH.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "FFmpeg conversion failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr}"
        )


def transcribe_wav(
    wav_path: Path,
    language: str = "it",
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Tuple[str, List[Dict], Dict]:
    """
    Returns:
      transcript_text: concatenated text
      segments_list: list of segments with start/end/text
      meta: language + some info
    """
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, info = model.transcribe(str(wav_path), language=language)

    segs: List[Dict] = []
    texts: List[str] = []
    for s in segments:
        segs.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text,
            }
        )
        texts.append(s.text.strip())

    transcript_text = " ".join([t for t in texts if t])

    meta = {
        "language": info.language,
        "model_name": model_name,
        "device": device,
        "compute_type": compute_type,
        "wav_path": str(wav_path),
    }
    return transcript_text, segs, meta


def save_outputs(out_dir: Path, transcript_text: str, segments: List[Dict], meta: Dict) -> None:
    ensure_dir(out_dir)

    (out_dir / "transcript.txt").write_text(transcript_text + "\n", encoding="utf-8")
    (out_dir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def asr_pipeline(input_audio_path: Path, session_dir: Path, language: str = "it", out_subdir: str = "out") -> Path:

    """
    Full pipeline:
      - create session folders
      - convert to wav
      - transcribe
      - save out/transcript.txt (+ json files)
    Returns path to transcript.txt
    """
    raw_dir = session_dir / "raw"
    wav_dir = session_dir / "wav"
    out_dir = session_dir / out_subdir
    ensure_dir(raw_dir)
    ensure_dir(wav_dir)
    ensure_dir(out_dir)

    # Convert to wav
    wav_path = wav_dir / "input.wav"
    run_ffmpeg_to_wav(input_audio_path, wav_path)

    # Transcribe
    transcript_text, segments, meta = transcribe_wav(wav_path, language=language)

    # Save
    save_outputs(out_dir, transcript_text, segments, meta)

    return out_dir / "transcript.txt"
