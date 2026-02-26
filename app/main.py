from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.paths import DATA_ROOT, SESSIONS_DIR, ensure_dirs
from app.db import init_db

app = FastAPI(title="AI Cognitive Screening Backend (MVP)")

ensure_dirs()

app.mount("/assets", StaticFiles(directory=str(DATA_ROOT)), name="assets")
app.mount("/files", StaticFiles(directory=str(SESSIONS_DIR)), name="files")

app.include_router(router)


@app.on_event("startup")
def on_startup():
    init_db()

    