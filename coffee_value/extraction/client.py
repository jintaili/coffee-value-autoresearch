"""Small HTTP adapter for the documented TypeSafe System One endpoint."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .questions import MODEL, QUESTIONS

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
INPUT_USD_PER_TOKEN = 0.042 / 1_000_000


class ProviderError(RuntimeError):
    pass


def evaluate(state: dict[str, str], *, questions: dict[str, dict] = QUESTIONS, model: str = MODEL,
             timeout: float = 60, attempts: int = 3) -> tuple[dict[str, Any], float, int]:
    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise ProviderError("TYPESAFE_API_KEY is not configured")
    body = json.dumps({"state": state, "model": model, "questions": questions}, separators=(",", ":")).encode()
    request = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "coffee-value-shared/0.1"})
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            return result, time.monotonic() - started, attempt
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (429, 529, 500, 502, 503, 504)
            if not retryable or attempt == attempts:
                raise ProviderError(f"TypeSafe HTTP {exc.code}") from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = min(30.0, max(0.0, float(retry_after))) if retry_after else float(2 ** (attempt - 1))
            except ValueError:
                delay = float(2 ** (attempt - 1))
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise ProviderError(f"TypeSafe connection failed: {type(exc).__name__}") from exc
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")
