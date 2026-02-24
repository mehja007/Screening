from __future__ import annotations

from fastapi import HTTPException

from app.tests_engine.demo_protocol import DEMO_PROTOCOL_V1
from app.repositories.db_repo import db_list_mmse_prompts, ensure_mmse_prompts_exist


def get_protocol_steps(protocol: str, lang: str = "it"):
    lang = (lang or "it").lower()
    if lang.startswith("it"):
        lang = "it"

    if protocol == "demo_v1":
        return DEMO_PROTOCOL_V1

    if protocol == "mmse_v1":
        ensure_mmse_prompts_exist(protocol="mmse_v1", lang=lang)
        rows = db_list_mmse_prompts(protocol="mmse_v1", lang=lang)

        # Step compatibile con il tuo codice (step_id, question)
        class _Step:
            def __init__(self, step_id: str, question: str):
                self.step_id = step_id
                self.question = question

        return [_Step(step_id=f"mmse_step{r.step:02d}", question=r.text) for r in rows]

    raise HTTPException(status_code=400, detail="Unknown protocol. Use protocol=demo_v1 or mmse_v1")