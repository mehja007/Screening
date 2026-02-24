from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

#funzione che restituisce data e ora in formato ISO
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionState:
    protocol: str
    current_step: int
    completed: bool

    # converte lo stato della sessione in un dizionario, utile per la serializzazione in JSON
    def to_dict(self) -> Dict[str, Any]: #to_dict è un metoodo che crea un dizionario con chiavi e valori corrispondenti agli attributi della classe SessionState
        return {
            "protocol": self.protocol,
            "current_step": self.current_step,
            "completed": self.completed,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SessionState":  #crea sessione da un dizionario 
        return SessionState( #costruisce e ritorna l'istanza
            protocol=str(d.get("protocol", "demo_v1")),
            current_step=int(d.get("current_step", 0)),
            completed=bool(d.get("completed", False)),
        )


def state_path(session_dir: Path) -> Path: #funzione utile per recuperare il percordo per state.json
    return session_dir / "state.json"


def turns_path(session_dir: Path) -> Path: #funzione utile per recuperare il percordo per turns.json
    return session_dir / "turns.json"


def load_state(session_dir: Path) -> SessionState: #funz che legge e carica lo stato della sessione da un file JSON
    p = state_path(session_dir) #assegna a p il percordso del file state.json
    if not p.exists():
        return SessionState(protocol="demo_v1", current_step=0, completed=False)
    return SessionState.from_dict(json.loads(p.read_text(encoding="utf-8")))
#altrimenti legge il JSON dal file, lo decodifica e costruisce un oggetto SessionState utilizzando il metodo from_dict


def save_state(session_dir: Path, state: SessionState) -> None: #funzione che salva lo stato della sessione in un file JSON
    state_path(session_dir).write_text( #apre e sovrascrive il file state.json 
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2), #converte lo stato in un dizionario comprensibile in JSON, lo serializza in una stringa JSON formattata e la scrive nel file
        encoding="utf-8",
    )


def load_turns(session_dir: Path) -> List[Dict[str, Any]]: #legge e ritorna la listsa dei turni
    p = turns_path(session_dir) #assegna a p il percorso del file turns.json
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")) #se esiste, legge il contenuto del file, lo decodifica da JSON e ritorna la lista dei turni


def append_turn(session_dir: Path, turn: Dict[str, Any]) -> None: #aggiunge un nuovo turno alla lista 
    turns = load_turns(session_dir) #carica la lista dei turni esistenti
    turns.append(turn) #aggiunge il nuovo turno alla lista
    turns_path(session_dir).write_text( #apre e sovrascrive il file turns.json con la lista aggiornata dei turni, serializzata in JSON
        json.dumps(turns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def basic_text_features(text: str) -> Dict[str, Any]: #funzione che calcola alcune caratteristiche di base del testo, come il numero di caratteri e parole, e ritorna queste informazioni in un dizionario
    tokens = [t for t in text.strip().split() if t] #tokenizza il testo, rimuovendo eventuali spazi bianchi all'inizio o alla fine e dividendo il testo in parole, filtrando eventuali token vuoti
    return {
        "char_count": len(text),
        "word_count": len(tokens),
    }


def make_turn(  #funzione che un oggetto turno
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
