"""Helpers for coaxing valid JSON out of free-form LLM responses."""

import json
from typing import Any


def extract_json_object(text: str) -> Any:
    """Extract and parse the first top-level JSON object from ``text``.

    LLMs frequently wrap query JSON in markdown code fences or surround it with
    explanatory prose. This strips ```/```json fences and slices from the first
    ``{`` to the last ``}`` before parsing.

    Raises:
        ValueError: if no ``{...}`` span is found.
        json.JSONDecodeError: if the extracted span is not valid JSON.
    """
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in text")
    return json.loads(cleaned[start:end])
