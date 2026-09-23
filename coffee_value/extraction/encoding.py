"""Strict probability encoding. Failed extraction is never encoded as missing evidence."""

from __future__ import annotations

import math

from .questions import CONTRACT_VERSION, MODEL, QUESTIONS, question_hash

ENCODING_VERSION = "probabilities-1"


def feature_names(*, questions: dict[str, dict] = QUESTIONS) -> list[str]:
    return [f"jev:{qid}:{option}" for qid, q in questions.items() for option in q["criteria"]]


def validate(record: dict, *, questions: dict[str, dict] = QUESTIONS,
             contract_version: str = CONTRACT_VERSION, questions_hash: str | None = None,
             model: str = MODEL) -> None:
    if record.get("status") != "complete" or record.get("contract_version") != contract_version:
        raise ValueError("incomplete or mismatched extraction contract")
    expected_hash = questions_hash if questions_hash is not None else question_hash()
    if record.get("question_hash") != expected_hash or record.get("requested_model") != model or record.get("resolved_model") != model:
        raise ValueError("question or model version mismatch")
    answers = record.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("missing or unexpected answers")
    for qid, question in questions.items():
        answer = answers[qid]
        options = set(question["criteria"])
        probs = answer.get("probabilities") if isinstance(answer, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice" or answer.get("choice") not in options or not isinstance(probs, dict) or set(probs) != options:
            raise ValueError(f"invalid answer for {qid}")
        if any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()):
            raise ValueError(f"invalid probabilities for {qid}")
        if abs(sum(probs.values()) - 1) > 0.015:
            raise ValueError(f"probabilities do not sum to one for {qid}")


def encode(record: dict, *, questions: dict[str, dict] = QUESTIONS,
           contract_version: str = CONTRACT_VERSION, questions_hash: str | None = None,
           model: str = MODEL) -> list[float]:
    validate(record, questions=questions, contract_version=contract_version,
             questions_hash=questions_hash, model=model)
    return [float(record["answers"][qid]["probabilities"][option]) for qid, q in questions.items() for option in q["criteria"]]
