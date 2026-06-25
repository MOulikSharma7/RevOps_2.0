"""Automation agent nodes (in development).

This currently performs a deterministic, dependency-free transformation: it
wraps the NBA remediation plan into a commented, non-executing script skeleton.
The intent is to later replace ``scaffold_script`` with an LLM that emits real,
runnable Python/Bash. Keeping it deterministic for now means the orchestrator
can exercise the full pipeline without an LLM and the behaviour is unit-testable.
"""

from .state import AutomationState

_HEADER = {
    "Linux": ("bash", "#!/usr/bin/env bash", "#"),
    "Windows": ("powershell", "# Requires -Version 5.1", "#"),
}


def scaffold_script(state: AutomationState) -> dict:
    target_os = state.get("target_os") or "Linux"
    nba_plan = (state.get("nba_plan") or "").strip() or "No remediation plan provided."

    script_type, shebang, comment = _HEADER.get(target_os, _HEADER["Linux"])

    commented_plan = "\n".join(f"{comment} {line}" for line in nba_plan.splitlines())
    script_content = (
        f"{shebang}\n"
        f"{comment} ---------------------------------------------------------------\n"
        f"{comment} AUTO-GENERATED REMEDIATION SCAFFOLD ({target_os})\n"
        f"{comment} automation_agent is in development: review before running.\n"
        f"{comment} The plan below is NOT yet translated into executable commands.\n"
        f"{comment} ---------------------------------------------------------------\n"
        f"{commented_plan}\n"
    )

    return {"script_content": script_content, "script_type": script_type}
