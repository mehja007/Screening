from __future__ import annotations

from pathlib import Path

from app.repositories.db_repo import db_add_message, db_get_mmse_prompt
from app.services.protocols import get_protocol_steps
from app.services.tts_qwen import QwenTTSService, TTSConfig

# Carica una volta sola e riusa
tts_service = QwenTTSService(TTSConfig())


def create_first_question(session_id: str, protocol: str, lang: str, session_dir: Path) -> dict:
    if protocol == "mmse_v1":
        p0 = db_get_mmse_prompt(protocol="mmse_v1", lang=lang, step=0)

        first_question = p0.text
        question_audio_url = f"/assets/{p0.audio_relpath}"

        db_add_message(session_id, "assistant", first_question, question_audio_url)

        return {
            "session_id": session_id,
            "protocol": protocol,
            "question": first_question,
            "question_audio_url": question_audio_url,
        }

    steps = get_protocol_steps(protocol)
    first_question = steps[0].question if steps else ""

    system_dir = session_dir / "out" / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    question_wav = system_dir / "question_step00.wav"
    tts_service.synthesize_to_wav(first_question, question_wav)

    question_audio_url = f"/files/{session_id}/out/system/question_step00.wav"
    db_add_message(session_id, "assistant", first_question, question_audio_url)

    return {
        "session_id": session_id,
        "protocol": protocol,
        "question": first_question,
        "question_audio_url": question_audio_url,
    }