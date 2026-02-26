import json
import urllib.request
from typing import Dict
import re

from datetime import datetime
from zoneinfo import ZoneInfo

WEEKDAY_IT = {
    0: "lunedì",
    1: "martedì",
    2: "mercoledì",
    3: "giovedì",
    4: "venerdì",
    5: "sabato",
    6: "domenica",
}

MONTH_IT = {
    1: "gennaio",
    2: "febbraio",
    3: "marzo",
    4: "aprile",
    5: "maggio",
    6: "giugno",
    7: "luglio",
    8: "agosto",
    9: "settembre",
    10: "ottobre",
    11: "novembre",
    12: "dicembre",
}

def season_it_meteorological(month: int) -> str:
    if month in (12, 1, 2):
        return "inverno"
    if month in (3, 4, 5):
        return "primavera"
    if month in (6, 7, 8):
        return "estate"
    return "autunno"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"


STEP_CONFIG: Dict[str, Dict] = {
    "0": {
        "max_score": 5,
        "rubric": (
            "Valuta le 5 informazioni temporali fornite dall'utente:\n"
            "- giorno del mese\n"
            "- mese\n"
            "- anno\n"
            "- giorno della settimana\n"
            "- stagione\n"
            "Assegna 1 punto per ciascuna informazione corretta."
        ),
    },
    "1": {
        "max_score": 3,
        "rubric": (
            "Valuta le 3 informazioni spaziali fornite dall'utente:\n"
            "- città\n"
            "- regione\n"
            "- stato\n"
            "Assegna 1 punto per ciascuna informazione corretta."
        ),
    },
    "2": {
        "max_score": 3,
        "rubric": (
            "Parole target: casa, pane, gatto.\n"
            "Assegna 1 punto per ogni parola correttamente ripetuta.\n"
            "L'ordine non è rilevante."
        ),
    },
    "3": {
        "max_score": 5,
        "rubric": (
            "L'utente deve contare all'indietro di 7 partendo da 100.\n"
            "Valuta i primi 5 risultati.\n"
            "Assegna 1 punto per ogni sottrazione corretta."
        ),
    },
    "4": {
        "max_score": 3,
        "rubric": (
            "Richiamo parole target: casa, pane, gatto.\n"
            "Assegna 1 punto per ogni parola correttamente ricordata.\n"
            "L'ordine non è rilevante."
        ),
    },
    "5": {
        "max_score": 1,
        "rubric": (
            "Frase target: 'tigre contro tigre'.\n"
            "Assegna 1 punto se la frase è ripetuta correttamente "
            "(tolleranza minima per piccoli errori ASR)."
        ),
    },
}

def _normalize_step_id(step_id) -> str:
    s = str(step_id).strip()

    # step00 / step01 / step05 -> 0..5
    m = re.fullmatch(r"step(\d+)", s)
    if m:
        return str(int(m.group(1)))  # int() rimuove zeri iniziali

    # "00" -> "0"
    if re.fullmatch(r"\d+", s):
        return str(int(s))

    return s


def score_with_ollama(step_id: str, question: str, answer: str) -> dict:
    step_key = _normalize_step_id(step_id)
    config = STEP_CONFIG.get(step_key)


    if not config:
        return {
            "score": 0,
            "max_score": 0,
            "reason": "step non configurato",
        }

    max_score = config["max_score"]
    rubric = config["rubric"]


        # Ground truth per orientamento temporale (step 0)
    if step_key == "0":
        dt = datetime.now(ZoneInfo("Europe/Rome"))
        rubric += (
            "\n\nVALORI CORRETTI (ground truth, timezone Europe/Rome):\n"
            f"- giorno del mese: {dt.day}\n"
            f"- mese: {MONTH_IT[dt.month]} ({dt.month})\n"
            f"- anno: {dt.year}\n"
            f"- giorno della settimana: {WEEKDAY_IT[dt.weekday()]}\n"
            f"- stagione (meteorologica): {season_it_meteorological(dt.month)}\n"
            "Assegna 1 punto per ciascun elemento correttamente indicato.\n"
            "Se l'utente fornisce più valori, considera corretto se include quello giusto.\n"
        )

    system_prompt = (
        "Sei un valutatore clinico del test MMSE.\n"
        f"Questo item vale massimo {max_score} punti.\n\n"
        "CRITERI DI VALUTAZIONE:\n"
        f"{rubric}\n\n"
        "Assegna un punteggio intero tra 0 e max_score.\n"
        "Rispondi SOLO con JSON valido nel formato:\n"
        "{\"score\": <int>, \"max_score\": <int>, \"reason\": \"...\"}\n"
        "Non aggiungere altro testo."
    )

    user_prompt = (
        f"step_id: {step_key}\n"
        f"domanda: {question}\n"
        f"risposta_utente: {answer}\n"
        "Restituisci SOLO JSON."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {
            "score": 0,
            "max_score": max_score,
            "reason": f"Errore chiamata LLM: {str(e)}",
        }

    content = (data.get("message") or {}).get("content", "").strip()

    try:
        out = json.loads(content)
    except Exception:
        return {
            "score": 0,
            "max_score": max_score,
            "reason": "LLM non ha restituito JSON valido",
        }




    if not isinstance(out, dict):
        return {
            "score": 0,
            "max_score": max_score,
            "reason": "Output LLM non valido",
        }

    score = out.get("score", 0)
    reason = out.get("reason", "nessuna motivazione")

    if not isinstance(score, int):
        score = 0

    # Clamp sicurezza
    score = max(0, min(score, max_score))

    return {
        "score": score,
        "max_score": max_score,
        "reason": reason,
    }