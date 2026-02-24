from app.db import SessionLocal, init_db
from app.models import PromptAssetDB

def mmse_prompts_it_spoken():
    return [
        (0, "Buongiorno. Iniziamo il test. Mi dica che giorno del mese è oggi, in che mese siamo, in che anno siamo, che giorno della settimana è, e in che stagione siamo."),
        (1, "Ora mi dica dove ci troviamo: in che città siamo, in che regione, e in che stato."),
        (2, "Adesso le dirò tre parole. Le ripeta subito dopo di me: casa, pane, gatto."),
        (3, "Ora conti all'indietro di sette partendo da cento. Mi dica i primi cinque risultati."),
        (4, "Ora mi ripeta le tre parole di prima."),
        (5, "Ripeta questa frase: tigre contro tigre."),
    ]

def main():
    init_db()

    db = SessionLocal()
    try:
        protocol = "mmse_v1"
        lang = "it"

        for step, text in mmse_prompts_it_spoken():
            audio_relpath = f"prompts/{protocol}/{lang}/step{step:02d}.wav"

            row = db.query(PromptAssetDB).filter_by(
                protocol=protocol,
                lang=lang,
                step=step
            ).one_or_none()

            if row is None:
                row = PromptAssetDB(
                    protocol=protocol,
                    lang=lang,
                    step=step,
                    text=text,
                    audio_relpath=audio_relpath
                )
                db.add(row)
            else:
                row.text = text
                row.audio_relpath = audio_relpath

        db.commit()
        print("Seed DB completato.")

    finally:
        db.close()


if __name__ == "__main__":
    main()