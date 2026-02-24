from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from app.services.tts_qwen import QwenTTSService, TTSConfig


def mmse_prompts_it() -> List[Tuple[int, str]]:
    """
    Versione MMSE solo parlato (no disegni, no oggetti fisici).
    step -> testo
    """
    return [
        (
            0,
            "Iniziamo il test. "
            "Mi dica che giorno del mese è oggi, in che mese siamo, in che anno siamo, "
            "che giorno della settimana è, e in che stagione siamo."
        ),
        (
            1,
            "Ora mi dica dove ci troviamo: in che città siamo, in che regione, "
            "e in che stato."
        ),
        (
            2,
            "Adesso le dirò tre parole. Le ripeta subito dopo di me: casa, pane, gatto."
        ),
        (
            3,
            "Ora conti all'indietro di sette partendo da cento. "
            "Mi dica i primi cinque risultati."
        ),
        (
            4,
            "Ora mi ripeta le tre parole di prima."
        ),
        (
            5,
            "Ripeta questa frase: tigre contro tigre."
        ),
    ]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "data" / "prompts" / "mmse_v1" / "it"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TTSConfig(
        speaker="Ryan",
        language="Auto",
        device_map="cpu",
    )

    tts = QwenTTSService(cfg)

    prompts = mmse_prompts_it()
    total = len(prompts)

    print(f"Output directory: {out_dir}")
    print(f"Generating {total} spoken MMSE prompts...")

    for i, (step, text) in enumerate(prompts, start=1):
        wav_path = out_dir / f"step{step:02d}.wav"

        if wav_path.exists() and wav_path.stat().st_size > 0:
            print(f"[{i:02d}/{total:02d}] step{step:02d}: already exists → skip")
            continue

        print(f"[{i:02d}/{total:02d}] step{step:02d}: generating...")
        tts.synthesize_to_wav(text=text, out_wav_path=wav_path)
        print(f"           saved: {wav_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()