from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Step:
    step_id: str
    question: str


MMSE_PROTOCOL_V1: List[Step] = [
    # 1) Orientamento nel tempo
    Step(
        step_id="orientation_time",
        question="Che giorno del mese è oggi? Che mese? Che anno? Che giorno della settimana? In che stagione siamo?",
    ),

    # 1b) Orientamento nel luogo
    Step(
        step_id="orientation_place",
        question="Dove ti trovi adesso? In che città? In che regione? In che stato? E a che piano (se lo sai)?",
    ),

    # 2) Registrazione: 3 parole
    Step(
        step_id="registration_3_words",
        question="Ora ti dirò tre parole: casa, pane, gatto. Ripetile subito, nell'ordine che preferisci.",
    ),

    # 3) Attenzione e calcolo
    Step(
        step_id="attention_serial_7s",
        question="Ora partiamo da 100 e sottrai 7 ogni volta. Dimmi i primi cinque risultati.",
    ),

    # Alternativa al calcolo
    Step(
        step_id="attention_mondo_backwards",
        question='Se ti è difficile fare i calcoli, dimmi la parola "MONDO" al contrario, una lettera alla volta.',
    ),

    # 4) Richiamo delle 3 parole
    Step(
        step_id="recall_3_words",
        question="Prima ti ho detto tre parole. Puoi ripeterle adesso?",
    ),

    # 5) Linguaggio - denominazione oggetti
    Step(
        step_id="language_naming_objects",
        question="Ora ti mostrerò due oggetti, una matita e un orologio. Come si chiamano?",
    ),

    # 5) Linguaggio - ripetizione frase
    Step(
        step_id="language_repeat_phrase",
        question='Ripeti questa frase: "TIGRE CONTRO TIGRE".',
    ),
]


DEMO_PROTOCOL_V1 = MMSE_PROTOCOL_V1
