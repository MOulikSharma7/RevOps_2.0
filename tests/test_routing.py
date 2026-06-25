"""Tests for the orchestrator's pure routing logic."""

import pytest

from orchestrator_agent.routing import AGENT_NODE_MAP, resolve_route


@pytest.mark.parametrize(
    "decision,expected_node",
    [
        ("db_agent", "db_node"),
        ("log_retriever_agent", "log_node"),
        ("RCA_agent", "rca_node"),
        ("NBA_agent", "nba_node"),
        ("automation_agent", "automation_node"),
        ("execute_agent", "execute_node"),
        ("RAG_agent", "rag_node"),
    ],
)
def test_known_agents_route_to_their_node(decision, expected_node):
    assert resolve_route(decision) == expected_node


def test_finish_returns_none():
    assert resolve_route("FINISH") is None


def test_unknown_decision_returns_none():
    # Fail safe: an unmappable decision should stop the graph, not loop.
    assert resolve_route("some_hallucinated_agent") is None
    assert resolve_route("") is None


def test_every_node_name_is_unique():
    nodes = list(AGENT_NODE_MAP.values())
    assert len(nodes) == len(set(nodes))
