"""Pure routing helpers for the orchestrator graph.

Kept free of LangGraph/LLM imports so the supervisor's routing decision can be
unit-tested in isolation. ``graph.py`` maps a ``None`` result to LangGraph's
``END`` sentinel.
"""

# Maps a supervisor decision (an AVAILABLE_AGENTS key, or "FINISH") to the
# corresponding graph node. Keys MUST stay in sync with
# ``orchestrator_agent.prompts.AVAILABLE_AGENTS``.
AGENT_NODE_MAP = {
    "db_agent": "db_node",
    "log_retriever_agent": "log_node",
    "RCA_agent": "rca_node",
    "NBA_agent": "nba_node",
    "automation_agent": "automation_node",
    "execute_agent": "execute_node",
    "RAG_agent": "rag_node",
}


def resolve_route(decision: str):
    """Return the target node name for a supervisor ``decision``.

    Returns ``None`` when the graph should terminate -- either because the
    supervisor explicitly chose ``FINISH`` or because ``decision`` could not be
    mapped to a known node (fail safe: stop rather than loop).
    """
    if decision == "FINISH":
        return None
    return AGENT_NODE_MAP.get(decision)
