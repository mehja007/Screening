from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from app.db import SessionLocal
from app.models import PromptAssetDB

from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException #FastAPI è il server
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse
import json

from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="AI Cognitive Screening Backend (MVP)")

APP_ROOT = Path(__file__).resolve().parents[1]  # root progetto

DATA_ROOT = APP_ROOT / "data"          # <-- cartella data/
SESSIONS_DIR = DATA_ROOT / "sessions"  # <-- cartella data/sessions/


SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/assets", StaticFiles(directory=str(DATA_ROOT)), name="assets")

from app.services.asr_whisper import asr_pipeline
from app.services.llm_ollama import score_with_ollama

from app.services.tts_qwen import QwenTTSService, TTSConfig #Necessario a rendere l'audio leggibile ovvero trasformandolo in WAV

from app.db import SessionLocal, init_db
from app.models import SessionDB, MessageDB


from app.core.session_state import (  #Dove avviene la gestione del stato, turno del paziente e turno del sistema nel colloquio
    SessionState,
    append_turn,
    load_state,
    save_state,
    make_turn,
)

from app.tests_engine.demo_protocol import DEMO_PROTOCOL_V1




@app.on_event("startup")
def on_startup():
    init_db()

# TTS service (carica il modello al primo uso e poi lo riusa), occhio non serve caricarlo ad ogni richiesta ma una volta sola
tts_service = QwenTTSService(TTSConfig())

# Rende scaricabili i file dentro data/sessions/<session_id>/...
app.mount("/files", StaticFiles(directory=str(SESSIONS_DIR)), name="files")

@app.get("/", response_class=PlainTextResponse)
def healthcheck():
    return "OK"

@app.get("/app")
def web_app():
    return FileResponse(APP_ROOT / "app" / "web" / "index.html")

def get_protocol_steps(protocol: str, lang: str = "it"):
    # Normalizza lingua (evita "it-IT", "IT", ecc.)
    lang = (lang or "it").lower()
    if lang.startswith("it"):
        lang = "it"

    if protocol == "demo_v1":
        return DEMO_PROTOCOL_V1

    if protocol == "mmse_v1":
        db = SessionLocal()
        try:
            rows = (
                db.query(PromptAssetDB)
                .filter_by(protocol="mmse_v1", lang=lang)
                .order_by(PromptAssetDB.step.asc())
                .all()
            )
        finally:
            db.close()

        if not rows:
            raise HTTPException(
                status_code=500,
                detail=f"Nessun prompt MMSE trovato nel DB: protocol=mmse_v1 lang={lang}",
            )

        # Oggetto step compatibile col tuo codice (step_id, question)
        class _Step:
            def __init__(self, step_id: str, question: str):
                self.step_id = step_id
                self.question = question

        return [_Step(step_id=f"mmse_step{r.step:02d}", question=r.text) for r in rows]

    raise HTTPException(status_code=400, detail="Unknown protocol. Use protocol=demo_v1 or mmse_v1")


def db_add_session(session_id: str, protocol: str):
    db = SessionLocal()
    try:
        db.add(SessionDB(id=session_id, protocol=protocol))
        db.commit()
    finally:
        db.close()


def db_add_message(session_id: str, role: str, text: str, audio_url: str | None = None):
    db = SessionLocal()
    try:
        db.add(MessageDB(session_id=session_id, role=role, text=text, audio_url=audio_url))
        db.commit()
    finally:
        db.close()

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
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

        # 👇 Questo lo rende leggibile (indentato)
        return JSONResponse(
            content=json.loads(json.dumps(data, indent=2))
        )

    finally:
        db.close()

@app.post("/sessions")
def create_session(protocol: str = "demo_v1", lang: str = "it"):
    session_id = uuid.uuid4().hex
    session_dir = SESSIONS_DIR  / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    state = SessionState(protocol=protocol, current_step=0, completed=False)
    save_state(session_dir, state)

    # DB: salva sessione
    db_add_session(session_id, protocol)

    # --- MMSE: prima domanda da DB + audio pre-generato ---
    if protocol == "mmse_v1":
        db = SessionLocal()
        try:
            p0 = (
                db.query(PromptAssetDB)
                .filter_by(protocol="mmse_v1", lang=lang, step=0)
                .one()
            )
        finally:
            db.close()

        first_question = p0.text
        question_audio_url = f"/assets/{p0.audio_relpath}"

        # DB: salva primo messaggio (domanda del sistema)
        db_add_message(session_id, "assistant", first_question, question_audio_url)

        return {
            "session_id": session_id,
            "protocol": protocol,
            "question": first_question,
            "question_audio_url": question_audio_url,
        }

    # --- fallback: comportamento attuale per demo_v1 (TTS runtime) ---
    steps = get_protocol_steps(protocol)
    first_question = steps[0].question if steps else ""

    system_dir = session_dir / "out" / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    question_wav = system_dir / "question_step00.wav"
    tts_service.synthesize_to_wav(first_question, question_wav)

    question_audio_url = f"/files/{session_id}/out/system/question_step00.wav"

    db_add_message(session_id, "assistant", first_question, question_audio_url)

    return {
        "session_id": session_id,
        "protocol": protocol,
        "question": first_question,
        "question_audio_url": question_audio_url,
    }

@app.post("/sessions/{session_id}/answer_audio")#endpoint per rispondere alla domanda del sistema con un audio,
# riceve l'audio dell'utente, lo salva, lo trascrive, aggiorna lo stato della sessione e sintetizza la prossima domanda del sistema


async def answer_audio(session_id: str, file: UploadFile = File(...), language: str = "it"):
    session_dir = SESSIONS_DIR  / session_id # creo un oggetto Path per il percorso della sessione specifica, dove saranno salvati l'audio dell'utente, la trascrizione e i metadata
    if not session_dir.exists(): # se il percorso della sessione non esiste, significa che la sessione non è stata creata, quindi ritorno un errore 404
        raise HTTPException(status_code=404, detail="Session not found. Create session first.")

    state = load_state(session_dir) #carico lo stato della sessione, che contiene informazioni sul protocollo in uso, lo step corrente e se la sessione è completata o meno
    if state.completed: # se la sessione è già completata, la funzione risponde con un JSON che indica che la sessione è già completata e non accetta ulteriori risposte audio, evitando di processare l'audio dell'utente o avanzare lo stato della sessione
        return {
            "session_id": session_id,
            "completed": True,
            "message": "Session already completed.",
        }

    steps = get_protocol_steps(state.protocol, lang=language) #ottengo la lista degli step del protocollo in uso, in base all'informazione salvata nello stato della sessione.

   
    if state.current_step >= len(steps): #se lo step corrente è maggiore o uguale al numero totale di step del protocollo
        state.completed = True 
        save_state(session_dir, state) #salva lo stato aggiornato
        return { #risponde con un JSON che indica ...
            "session_id": session_id,
            "completed": True,
            "message": "No more steps.",
        }

    step = steps[state.current_step] # assegno lo step corrente alla variabile step 
    #che contiene informazioni come step_id, question 

    # 1) Salva audio dell'utente (legato allo step)
    raw_dir = session_dir / "raw" # crea Path per la cartella raw, dove saranno salvati gli audio originali dell'utente
    raw_dir.mkdir(parents=True, exist_ok=True) #assicura che la cartella esista

    suffix = Path(file.filename).suffix.lower() or ".bin" #ottiene l'estensione del file caricato, se non è presente assegna .bin come estensione di default. 
    input_path = raw_dir / f"step{state.current_step:02d}_{step.step_id}{suffix}"
    #nome del file audio, che include step corrente, step_id e estensione del file 
    try:
        with input_path.open("wb") as f: #apre il file in modalità scrittura binaria
            shutil.copyfileobj(file.file, f) #copia il contenuto del file caricato(file.file) nel file aperto (f), 
            #salvando così l'audio dell'utente nella posizione specificata da input_path
    finally:
        await file.close() #garantisce che lo stream venga chiuso anche in caso di eccezione

    # 2) ASR pipeline -> out/stepXX_<step_id>/
    try: #per gestire eventuali errori
        step_out = f"out/step{state.current_step:02d}_{step.step_id}" #crea il percorso di output per lo step corrente
        transcript_path = asr_pipeline( #chiama la funzione asr_pipelineda app.services.asr_whisper, 
            input_path, #inizia l'invocazione che ritorna percorso del file di trascrizione
            session_dir=session_dir, # passa il percorso della sessione per salvare i risultati
            language=language, 
            out_subdir=step_out, # specifica la sottocartella di output per questo step
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) #Intercetta qualsiasi eccezione sollevata dall'ASR

    transcript_text = transcript_path.read_text(encoding="utf-8").strip()
    #legge il contenuto del file di trascrizione generato dall'ASR, lo decodifica come testo e rimuove eventuali spazi bianchi all'inizio o alla fine, ottenendo così la trascrizione testuale dell'audio dell'utente.
    
    out_dir = session_dir / step_out # è il path completo della cartella di output per lo step corrente

    # --- LLM scoring (solo registration_3_words e recall_3_words) ---
    llm_score = score_with_ollama(step.step_id, step.question, transcript_text)

    # salva su file accanto agli output dello step
    (out_dir / "llm_score.json").write_text(
        json.dumps(llm_score, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


    
    # 3) Salva il turno
    segments_path = out_dir / "segments.json" #crea un percorso per il file segments.json, che conterrà informazioni sui segmenti di audio riconosciuti dall'ASR, come timestamp e testo associato a ciascun segmento.
    meta_path = out_dir / "meta.json"  #crea un percorso per il file meta.json, che conterrà informazioni aggiuntive sul turno, come ad esempio la durata dell'audio, la confidenza dell'ASR, o altre metriche rilevanti per l'analisi del colloquio.

#make_turn è una funzione che crea un oggetto turno, che rappresenta l'interazione tra l'utente e il sistema in uno specifico step del protocollo
    turn = make_turn(
        step_index=state.current_step,
        step_id=step.step_id,
        question=step.question,
        transcript=transcript_text,
        audio_path=str(input_path),
        transcript_path=str(transcript_path),
        segments_path=str(segments_path),
        meta_path=str(meta_path),
    )
    append_turn(session_dir, turn) #aggiunge il turno creato alla sessione 

    # 4) Avanza stato
    state.current_step += 1  #incrementa indice dello step corrente nella sessione
    if state.current_step >= len(steps): #controlla se lo step corrente ha superato il numero totale di step del protocollo, 
        state.completed = True # se sì, imposta lo stato della sessione come completato
    save_state(session_dir, state) # salva lo stato aggiornato della sessione

    next_question = None
    next_audio_url = None

    if not state.completed:
        if state.protocol == "mmse_v1":
            db = SessionLocal()
            try:
                nxt = (
                    db.query(PromptAssetDB)
                    .filter_by(protocol="mmse_v1", lang=language, step=state.current_step)
                    .one()
                )
            finally:
                db.close()

            next_question = nxt.text
            next_audio_url = f"/assets/{nxt.audio_relpath}"
        else:
            next_question = steps[state.current_step].question

        
    # 5) TTS della risposta del sistema (prossima domanda o chiusura)
    system_text = next_question if next_question else "Grazie. Il test è terminato."

    if state.protocol == "mmse_v1":
        # audio già pronto solo se c'è una prossima domanda
        reply_audio_url = next_audio_url
    else:
        reply_wav_path = out_dir / "system_reply.wav"
        tts_service.synthesize_to_wav(system_text, reply_wav_path)
        reply_audio_url = f"/files/{session_id}/{step_out}/system_reply.wav"


    # DB: salva messaggio utente (trascrizione + audio registrato)
    user_audio_url = f"/files/{session_id}/raw/{input_path.name}"
    db_add_message(session_id, "user", transcript_text, user_audio_url)

    # DB: salva messaggio sistema (testo + audio TTS)
    db_add_message(session_id, "assistant", system_text, reply_audio_url)

    return {
        "session_id": session_id,
        "protocol": state.protocol,
        "step_answered": step.step_id,
        "transcript": transcript_text,
        "completed": state.completed,
        "next_question": next_question,
        "system_text": system_text,
        "reply_audio_url": reply_audio_url,
        "llm_score": llm_score,
    }
