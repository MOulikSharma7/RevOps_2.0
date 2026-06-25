"""Shared state for the Orchestrator (supervisor) graph.

``messages`` accumulates across nodes via ``operator.add`` so every sub-agent
can append its ``OUTPUT from <agent>`` turn to the running conversation that the
supervisor reasons over.
"""

import operator
from typing import Annotated, List, TypedDict


class IntentState(TypedDict, total=False):
    # Running conversation. Each item is {"role": "user"|"assistant", "content": str}.
    # The reducer appends new messages rather than overwriting the list.
    messages: Annotated[List[dict], operator.add]

    # Populated by the retriever node (semantic shortlist of candidate agents).
    possible_agents: List[str]
    agent_descriptions: str

    # Populated by the supervisor node; drives conditional routing.
    selected_agent: str
