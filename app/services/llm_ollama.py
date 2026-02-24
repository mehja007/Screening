import json
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"


def score_with_ollama(step_id: str, question: str, answer: str) -> dict:
    # Valutiamo SOLO questi 2 step per ora
    if step_id not in ("registration_3_words", "recall_3_words"):
        return {"score": None, "max_score": None, "reason": "step non valutato (MVP)"}

    max_score = 3

    system = (
        "Sei un valutatore del test MMSE.\n"
        f"Questo item vale massimo {max_score} punti.\n"
        "Valuta quante delle parole target sono presenti nella risposta dell'utente.\n"
        "Parole target: casa, pane, gatto.\n"
        "Assegna un punteggio intero 0..3.\n"
        "Rispondi SOLO con JSON valido nel formato:\n"
        "{\"score\": <int>, \"max_score\": <int>, \"reason\": \"...\"}\n"
        "Non aggiungere altro testo."
    )

    user = (
        f"step_id: {step_id}\n"
        f"domanda: {question}\n"
        f"risposta: {answer}\n"
        "Restituisci SOLO JSON."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = (data.get("message") or {}).get("content", "").strip()

    try:
        out = json.loads(content)
    except Exception:
        out = {
            "score": None,
            "max_score": max_score,
            "reason": "LLM non ha restituito JSON valido",
        }

    return out
