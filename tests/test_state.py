"""Tests for the orchestrator IntentState contract."""

import operator
import typing

from orchestrator_agent.state import IntentState


def test_intent_state_has_expected_fields():
    hints = typing.get_type_hints(IntentState, include_extras=True)
    assert {"messages", "possible_agents", "agent_descriptions", "selected_agent"} <= set(hints)


def test_messages_field_uses_additive_reducer():
    # LangGraph relies on the operator.add reducer to *append* each agent's
    # output to the running conversation rather than overwrite it.
    hints = typing.get_type_hints(IntentState, include_extras=True)
    messages_hint = hints["messages"]
    metadata = getattr(messages_hint, "__metadata__", ())
    assert operator.add in metadata
