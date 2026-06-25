"""Tests for the placeholder execute/RAG sub-agents."""

import agents as dummy_agents


def test_execute_agent_is_clearly_simulated_and_runs_nothing():
    out = dummy_agents.execute_agent("#!/bin/bash\nsystemctl restart nginx")
    assert isinstance(out, str)
    assert "SIMULATED" in out


def test_rag_agent_echoes_query_as_simulated():
    out = dummy_agents.RAG_agent("How do I restart the gateway?")
    assert isinstance(out, str)
    assert "SIMULATED" in out
    assert "gateway" in out
