from __future__ import annotations

from datetime import datetime
from sqlalchemy import String, DateTime, Integer, Text, Column, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SessionDB(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # session_id
    protocol: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MessageDB(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    role: Mapped[str] = mapped_column(String, nullable=False)  # "assistant" / "user"
    text: Mapped[str] = mapped_column(Text, nullable=False)

    audio_url: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PromptAssetDB(Base):
    __tablename__ = "prompt_assets"

    id = Column(Integer, primary_key=True, index=True)
    protocol = Column(String(64), nullable=False)
    lang = Column(String(8), nullable=False)
    step = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    audio_relpath = Column(String(512), nullable=False)

    __table_args__ = (
        UniqueConstraint("protocol", "lang", "step", name="uq_prompt_assets_protocol_lang_step"),
    )