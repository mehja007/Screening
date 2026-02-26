from __future__ import annotations

from pathlib import Path

from app.repositories.db_repo import db_add_message, db_get_mmse_prompt


def create_first_question(session_id: str, lang: str, session_dir: Path) -> dict:
    # MMSE: prima domanda da DB + audio pre-generato
    p0 = db_get_mmse_prompt(protocol="mmse_v1", lang=lang, step=0)

    first_question = p0.text
    question_audio_url = f"/assets/{p0.audio_relpath}"

    db_add_message(session_id, "assistant", first_question, question_audio_url)

    return {
        "session_id": session_id,
        "protocol": "mmse_v1",
        "question": first_question,
        "question_audio_url": question_audio_url,
    }