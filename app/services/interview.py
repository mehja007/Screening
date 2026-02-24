from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import UploadFile, HTTPException

from app.core.session_state import append_turn, load_state, save_state, make_turn
from app.services.asr_whisper import asr_pipeline
from app.services.llm_ollama import score_with_ollama
from app.services.tts_qwen import QwenTTSService, TTSConfig
from app.services.protocols import get_protocol_steps
from app.repositories.db_repo import db_add_message, db_get_mmse_prompt
from app.core.paths import SESSIONS_DIR

# Carica una volta sola e riusa
tts_service = QwenTTSService(TTSConfig())




async def handle_answer_audio(session_id: str, file: UploadFile, language: str = "it") -> dict:
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found. Create session first.")

    state = load_state(session_dir)
    if state.completed:
        return {"session_id": session_id, "completed": True, "message": "Session already completed."}

    steps = get_protocol_steps(state.protocol, lang=language)

    if state.current_step >= len(steps):
        state.completed = True
        save_state(session_dir, state)
        return {"session_id": session_id, "completed": True, "message": "No more steps."}

    step = steps[state.current_step]

    # 1) Salva audio utente
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower() or ".bin"
    input_path = raw_dir / f"step{state.current_step:02d}_{step.step_id}{suffix}"
    try:
        with input_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    # 2) ASR
    try:
        step_out = f"out/step{state.current_step:02d}_{step.step_id}"
        transcript_path = asr_pipeline(
            input_path,
            session_dir=session_dir,
            language=language,
            out_subdir=step_out,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    transcript_text = transcript_path.read_text(encoding="utf-8").strip()
    out_dir = session_dir / step_out

    # 3) Scoring LLM
    llm_score = score_with_ollama(step.step_id, step.question, transcript_text)
    (out_dir / "llm_score.json").write_text(
        json.dumps(llm_score, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) Salva turno
    segments_path = out_dir / "segments.json"
    meta_path = out_dir / "meta.json"

    turn = make_turn(
        step_index=state.current_step,
        step_id=step.step_id,
        question=step.question,
        transcript=transcript_text,
        audio_path=str(input_path),
        transcript_path=str(transcript_path),
        segments_path=str(segments_path),
        meta_path=str(meta_path),
    )
    append_turn(session_dir, turn)

    # 5) Avanza stato
    state.current_step += 1
    if state.current_step >= len(steps):
        state.completed = True
    save_state(session_dir, state)

    next_question = None
    next_audio_url = None

    if not state.completed:
        if state.protocol == "mmse_v1":
            nxt = db_get_mmse_prompt(protocol="mmse_v1", lang=language, step=state.current_step)
            next_question = nxt.text
            next_audio_url = f"/assets/{nxt.audio_relpath}"
        else:
            next_question = steps[state.current_step].question

    # 6) Audio risposta sistema
    system_text = next_question if next_question else "Grazie. Il test è terminato."

    if state.protocol == "mmse_v1":
        reply_audio_url = next_audio_url
    else:
        reply_wav_path = out_dir / "system_reply.wav"
        tts_service.synthesize_to_wav(system_text, reply_wav_path)
        reply_audio_url = f"/files/{session_id}/{step_out}/system_reply.wav"

    # 7) Persistenza messaggi
    user_audio_url = f"/files/{session_id}/raw/{input_path.name}"
    db_add_message(session_id, "user", transcript_text, user_audio_url)
    db_add_message(session_id, "assistant", system_text, reply_audio_url)

    return {
        "session_id": session_id,
        "protocol": state.protocol,
        "step_answered": step.step_id,
        "transcript": transcript_text,
        "completed": state.completed,
        "next_question": next_question,
        "system_text": system_text,
        "reply_audio_url": reply_audio_url,
        "llm_score": llm_score,
    }