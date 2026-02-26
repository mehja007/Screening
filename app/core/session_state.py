from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionState:
    protocol: str
    current_step: int
    completed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "current_step": self.current_step,
            "completed": self.completed,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SessionState":
        return SessionState(
            protocol=str(d.get("protocol", "mmse_v1")),
            current_step=int(d.get("current_step", 0)),
            completed=bool(d.get("completed", False)),
        )


def state_path(session_dir: Path) -> Path:
    return session_dir / "state.json"


def turns_path(session_dir: Path) -> Path:
    return session_dir / "turns.json"


def load_state(session_dir: Path) -> SessionState:
    p = state_path(session_dir)
    if not p.exists():
        return SessionState(protocol="mmse_v1", current_step=0, completed=False)
    return SessionState.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_state(session_dir: Path, state: SessionState) -> None:
    state_path(session_dir).write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_turns(session_dir: Path) -> List[Dict[str, Any]]:
    p = turns_path(session_dir)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def append_turn(session_dir: Path, turn: Dict[str, Any]) -> None:
    turns = load_turns(session_dir)
    turns.append(turn)
    turns_path(session_dir).write_text(
        json.dumps(turns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def basic_text_features(text: str) -> Dict[str, Any]:
    tokens = [t for t in text.strip().split() if t]
    return {
        "char_count": len(text),
        "word_count": len(tokens),
    }


def make_turn(
    step_index: int,
    step_id: str,
    question: str,
    transcript: str,
    audio_path: str,
    transcript_path: str,
    segments_path: str,
    meta_path: str,
) -> Dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "step_index": step_index,
        "step_id": step_id,
        "question": question,
        "transcript": transcript,
        "features": basic_text_features(transcript),
        "artifacts": {
            "audio_path": audio_path,
            "transcript_path": transcript_path,
            "segments_path": segments_path,
            "meta_path": meta_path,
        },
    }