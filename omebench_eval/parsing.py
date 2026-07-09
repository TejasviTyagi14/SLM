"""Parse model text output into a list of mechanism steps."""

from __future__ import annotations

import json
import re
from typing import List, Optional


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove the opening fence (optionally with a language tag) and closing fence.
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_mechanism(raw: str) -> Optional[List[dict]]:
    """Best-effort extraction of the JSON mechanism list from model output.

    Handles: [ANSWER]...[/ANSWER] tags (cot prompt), markdown code fences,
    dict wrappers with a "mechanism" field, and raw JSON arrays embedded in prose.
    Returns a list of step dicts, or None if nothing parseable is found.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        inner = raw.get("mechanism", raw)
        return inner if isinstance(inner, list) else None
    if not isinstance(raw, str):
        return None

    text = raw

    # Prefer content inside [ANSWER]...[/ANSWER] if present.
    if "[ANSWER]" in text:
        after = text.split("[ANSWER]", 1)[1]
        text = after.split("[/ANSWER]", 1)[0] if "[/ANSWER]" in after else after

    text = _strip_code_fences(text)

    parsed = _try_json(text)
    if parsed is None:
        # Fall back to the first bracketed JSON array anywhere in the text.
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            parsed = _try_json(match.group(0))

    if parsed is None:
        return None
    if isinstance(parsed, dict):
        parsed = parsed.get("mechanism", parsed)
    return parsed if isinstance(parsed, list) else None


def _try_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None
