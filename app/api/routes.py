from __future__ import annotations

import uuid

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from app.core.paths import APP_ROOT, SESSIONS_DIR
from app.core.session_state import SessionState, save_state
from app.repositories.db_repo import db_add_session, db_get_messages_json
from app.services.interview import handle_answer_audio
from app.services.session_bootstrap import create_first_question
router = APIRouter()


@router.get("/", response_class=PlainTextResponse)
def healthcheck():
    return "OK"


@router.get("/app")
def web_app():
    return FileResponse(APP_ROOT / "app" / "web" / "index.html")


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    return db_get_messages_json(session_id)


@router.post("/sessions")
def create_session(lang: str = "it"):
    protocol = "mmse_v1"

    session_id = uuid.uuid4().hex
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    state = SessionState(protocol=protocol, current_step=0, completed=False)
    save_state(session_dir, state)

    db_add_session(session_id, protocol)

    return create_first_question(
        session_id=session_id,
        lang=lang,
        session_dir=session_dir,
    )

@router.post("/sessions/{session_id}/answer_audio")
async def answer_audio(session_id: str, file: UploadFile = File(...), language: str = "it"):
    return await handle_answer_audio(session_id=session_id, file=file, language=language)