from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


@dataclass
class TTSConfig:
    # modello più leggero per CPU (puoi cambiare dopo)
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    device_map: str = "cpu"
    dtype: torch.dtype = torch.float32

    # voce: puoi cambiarla in base a quelle disponibili
    speaker: str = "Ryan"
    language: str = "Auto"


class QwenTTSService:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self._model: Optional[Qwen3TTSModel] = None
        self._lock = Lock()

    def _load(self) -> Qwen3TTSModel:
        if self._model is None:
            self._model = Qwen3TTSModel.from_pretrained(
                self.cfg.model_id,
                device_map=self.cfg.device_map,
                dtype=self.cfg.dtype,
            )  
        return self._model

    def synthesize_to_wav(self, text: str, out_wav_path: Path, language: Optional[str] = None, speaker: Optional[str] = None) -> Path:
        out_wav_path.parent.mkdir(parents=True, exist_ok=True)

        lang = language or self.cfg.language
        spk = speaker or self.cfg.speaker

        with self._lock:
            model = self._load()
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=lang,
                speaker=spk,
                instruct="",
            )
            sf.write(str(out_wav_path), wavs[0], sr)

        return out_wav_path
