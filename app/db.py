from __future__ import annotations

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

APP_ROOT = Path(__file__).resolve().parents[1]  # .../ai-cognitive-screening
DB_PATH = APP_ROOT / "data" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # necessario con FastAPI
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def init_db():
    # importa modelli prima di create_all
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
