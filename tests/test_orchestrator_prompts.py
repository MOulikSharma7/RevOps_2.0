"""Tests for the orchestrator prompt/config and its consistency with routing."""

from orchestrator_agent.prompts import AVAILABLE_AGENTS, SUPERVISOR_SYSTEM_PROMPT
from orchestrator_agent.routing import AGENT_NODE_MAP


def test_available_agents_has_expected_keys():
    expected = {
        "db_agent",
        "log_retriever_agent",
        "RCA_agent",
        "NBA_agent",
        "automation_agent",
        "execute_agent",
        "RAG_agent",
    }
    assert set(AVAILABLE_AGENTS) == expected


def test_routing_map_matches_available_agents():
    # Regression guard: every advertised agent must be routable, and the router
    # must not reference agents the supervisor cannot choose.
    assert set(AGENT_NODE_MAP) == set(AVAILABLE_AGENTS)


def test_agent_descriptions_are_nonempty_strings():
    for name, desc in AVAILABLE_AGENTS.items():
        assert isinstance(desc, str) and desc.strip(), name


def test_supervisor_prompt_formats_and_demands_action_line():
    rendered = SUPERVISOR_SYSTEM_PROMPT.format(agent_list="- db_agent: does things")
    assert "db_agent: does things" in rendered
    assert "ACTION:" in rendered
