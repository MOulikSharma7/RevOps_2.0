"""Tests for the (in-development) automation agent's deterministic scaffolding."""

import pytest

from automation_agent.nodes import scaffold_script


def test_scaffold_wraps_plan_for_linux_by_default():
    result = scaffold_script({"nba_plan": "Step 1: restart nginx\nStep 2: tail logs"})
    assert result["script_type"] == "bash"
    assert result["script_content"].startswith("#!/usr/bin/env bash")
    # The plan content is preserved (as comments) rather than silently dropped.
    assert "Step 1: restart nginx" in result["script_content"]
    assert "Step 2: tail logs" in result["script_content"]


def test_scaffold_targets_windows_powershell():
    result = scaffold_script({"nba_plan": "Restart-Service nginx", "target_os": "Windows"})
    assert result["script_type"] == "powershell"
    assert "Restart-Service nginx" in result["script_content"]


def test_scaffold_handles_missing_plan():
    result = scaffold_script({})
    assert "No remediation plan provided." in result["script_content"]


def test_automation_graph_builds_if_langgraph_available():
    pytest.importorskip("langgraph")
    from automation_agent.graph import build_automation_graph

    app = build_automation_graph()
    out = app.invoke({"nba_plan": "Step 1: do thing", "target_os": "Linux"})
    assert "Step 1: do thing" in out["script_content"]
    assert out["script_type"] == "bash"
