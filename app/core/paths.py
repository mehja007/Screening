from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]  # root progetto
DATA_ROOT = APP_ROOT / "data"
SESSIONS_DIR = DATA_ROOT / "sessions"


def ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)