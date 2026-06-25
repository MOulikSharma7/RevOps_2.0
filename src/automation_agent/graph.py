from langgraph.graph import END, START, StateGraph

from .nodes import scaffold_script
from .state import AutomationState


def build_automation_graph():
    builder = StateGraph(AutomationState)

    builder.add_node("scaffold_script", scaffold_script)

    builder.add_edge(START, "scaffold_script")
    builder.add_edge("scaffold_script", END)

    return builder.compile()
