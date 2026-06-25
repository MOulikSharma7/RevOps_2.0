"""Lightweight placeholder sub-agents.

The orchestrator references ``execute_agent`` (command execution) and
``RAG_agent`` (knowledge-base lookup). Neither is implemented yet; these stubs
return a clearly-labelled simulated response so the supervisor graph can route
to them without crashing.
"""


def execute_agent(remediation_script: str) -> str:
    """Stub command executor. Does NOT run anything -- returns a simulated result."""
    preview = remediation_script.strip().splitlines()
    first_line = preview[0] if preview else "(empty)"
    return (
        "[SIMULATED EXECUTION] execute_agent is not yet implemented. "
        f"Received a remediation script ({len(remediation_script)} chars, "
        f"first line: {first_line!r}). No commands were run."
    )


def RAG_agent(query: str) -> str:
    """Stub knowledge-base agent. Returns a simulated 'no results' response."""
    return (
        "[SIMULATED RAG] RAG_agent is not yet implemented. "
        f"Would search SOPs/manuals for: {query!r}."
    )
