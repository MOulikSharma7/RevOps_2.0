from typing import TypedDict


class AutomationState(TypedDict, total=False):
    # Inputs from the orchestrator.
    nba_plan: str
    target_os: str

    # Outputs consumed by the orchestrator / execute_agent.
    script_content: str
    script_type: str
