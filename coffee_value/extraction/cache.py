"""Content-addressed extraction records with atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .questions import CONTRACT_VERSION, MODEL, question_hash
from .state import ADAPTER_VERSION


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def cache_key(state: dict[str, str], *, contract_version: str = CONTRACT_VERSION,
              questions_hash: str | None = None, model: str = MODEL) -> str:
    return digest({"state": state, "adapter": ADAPTER_VERSION, "contract": contract_version,
                   "questions": questions_hash if questions_hash is not None else question_hash(), "model": model})


def record_path(cache_dir: Path, state: dict[str, str], *, contract_version: str = CONTRACT_VERSION,
                questions_hash: str | None = None, model: str = MODEL) -> Path:
    key = cache_key(state, contract_version=contract_version, questions_hash=questions_hash, model=model)
    return cache_dir / key[:2] / f"{key}.json"


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".extract-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
