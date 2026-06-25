"""Tests for the LLM-response JSON extraction helper."""

import json

import pytest

from db_agent.json_utils import extract_json_object


def test_plain_json_object():
    assert extract_json_object('{"collection": "servers"}') == {"collection": "servers"}


def test_strips_markdown_json_fence():
    raw = '```json\n{"collection": "faults", "query": {"Status": "Active"}}\n```'
    assert extract_json_object(raw) == {"collection": "faults", "query": {"Status": "Active"}}


def test_strips_plain_code_fence():
    raw = '```\n{"a": 1}\n```'
    assert extract_json_object(raw) == {"a": 1}


def test_ignores_surrounding_prose():
    raw = 'Sure! Here is the query you asked for:\n{"collection": "x"}\nHope that helps.'
    assert extract_json_object(raw) == {"collection": "x"}


def test_handles_nested_objects():
    raw = '{"collection": "s", "query": {"Name": {"$regex": "beta", "$options": "i"}}}'
    parsed = extract_json_object(raw)
    assert parsed["query"]["Name"]["$regex"] == "beta"


def test_raises_when_no_object_present():
    with pytest.raises(ValueError):
        extract_json_object("there is no json here at all")


def test_raises_on_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        extract_json_object('{"collection": "servers",}')  # trailing comma
