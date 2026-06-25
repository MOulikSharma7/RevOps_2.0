"""Tests for the DB agent prompt templates."""

from db_agent.prompts import GENERATE_QUERY_PROMPT, TRANSFORM_PROMPT


def test_transform_prompt_formats_with_schema():
    rendered = TRANSFORM_PROMPT.format(schema_context="- Collection: 'servers'")
    assert "- Collection: 'servers'" in rendered


def test_generate_query_prompt_formats_with_both_placeholders():
    rendered = GENERATE_QUERY_PROMPT.format(
        schema_context="- Collection: 'servers'",
        clean_question="Find the IP for beta-02",
    )
    assert "Find the IP for beta-02" in rendered
    assert "- Collection: 'servers'" in rendered


def test_generate_query_prompt_keeps_json_braces_literal():
    # The template uses doubled braces for the literal JSON example; after a
    # format() call they must collapse to single braces (valid for the LLM).
    rendered = GENERATE_QUERY_PROMPT.format(schema_context="x", clean_question="y")
    assert '"collection"' in rendered
    assert "{{" not in rendered
