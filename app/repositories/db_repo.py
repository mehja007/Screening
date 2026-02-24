from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse
import json

from app.db import SessionLocal
from app.models import SessionDB, MessageDB, PromptAssetDB


def db_add_session(session_id: str, protocol: str) -> None:
    db = SessionLocal()
    try:
        db.add(SessionDB(id=session_id, protocol=protocol))
        db.commit()
    finally:
        db.close()


def db_add_message(session_id: str, role: str, text: str, audio_url: str | None = None) -> None:
    db = SessionLocal()
    try:
        db.add(MessageDB(session_id=session_id, role=role, text=text, audio_url=audio_url))
        db.commit()
    finally:
        db.close()


def db_get_messages_json(session_id: str) -> JSONResponse:
    db = SessionLocal()
    try:
        rows = (
            db.query(MessageDB)
            .filter(MessageDB.session_id == session_id)
            .order_by(MessageDB.id.asc())
            .all()
        )

        data = {
            "session_id": session_id,
            "messages": [
                {
                    "id": r.id,
                    "role": r.role,
                    "text": r.text,
                    "audio_url": r.audio_url,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ],
        }

        # JSON leggibile (indentato), stesso comportamento di prima
        return JSONResponse(content=json.loads(json.dumps(data, indent=2)))
    finally:
        db.close()


def db_get_mmse_prompt(protocol: str, lang: str, step: int) -> PromptAssetDB:
    db = SessionLocal()
    try:
        row = (
            db.query(PromptAssetDB)
            .filter_by(protocol=protocol, lang=lang, step=step)
            .one()
        )
        return row
    finally:
        db.close()


def db_list_mmse_prompts(protocol: str, lang: str) -> list[PromptAssetDB]:
    db = SessionLocal()
    try:
        rows = (
            db.query(PromptAssetDB)
            .filter_by(protocol=protocol, lang=lang)
            .order_by(PromptAssetDB.step.asc())
            .all()
        )
        return rows
    finally:
        db.close()


def ensure_mmse_prompts_exist(protocol: str, lang: str) -> None:
    rows = db_list_mmse_prompts(protocol=protocol, lang=lang)
    if not rows:
        raise HTTPException(
            status_code=500,
            detail=f"Nessun prompt MMSE trovato nel DB: protocol={protocol} lang={lang}",
        )