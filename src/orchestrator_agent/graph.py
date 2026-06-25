from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # in-memory checkpointer

from orchestrator_agent.state import IntentState
from orchestrator_agent.routing import resolve_route
from orchestrator_agent import nodes


def route_decision(state: IntentState) -> str:
    """Translate the supervisor's decision into a graph node (or END)."""
    decision = state["selected_agent"]
    next_node = resolve_route(decision)

    if next_node is None:
        if decision != "FINISH":
            print(f"WARNING: Could not map decision '{decision}' to a node. Defaulting to END.")
        else:
            print("DEBUG: Routing to END")
        return END

    print(f"DEBUG: Routing to '{next_node}'")
    return next_node


def build_intent_graph():
    workflow = StateGraph(IntentState)

    workflow.add_node("retriever_node", nodes.retrieve_agents_node)
    workflow.add_node("supervisor_node", nodes.supervisor_node)
    workflow.add_node("db_node", nodes.run_db_agent)
    workflow.add_node("log_node", nodes.run_log_retriever_agent)
    workflow.add_node("rca_node", nodes.run_rca_agent)
    workflow.add_node("nba_node", nodes.run_nba_agent)
    workflow.add_node("automation_node", nodes.run_automation_agent)
    workflow.add_node("execute_node", nodes.run_execute_agent)
    workflow.add_node("rag_node", nodes.run_rag_agent)

    workflow.set_entry_point("retriever_node")
    workflow.add_edge("retriever_node", "supervisor_node")

    workflow.add_conditional_edges(
        "supervisor_node",
        route_decision,
        {
            "db_node": "db_node",
            "log_node": "log_node",
            "rca_node": "rca_node",
            "nba_node": "nba_node",
            "automation_node": "automation_node",
            "execute_node": "execute_node",
            "rag_node": "rag_node",
            END: END,
        },
    )

    agents = ["db_node", "log_node", "rca_node", "nba_node", "automation_node", "execute_node", "rag_node"]
    for agent_node in agents:
        workflow.add_edge(agent_node, "supervisor_node")

    # --- MEMORY CHECKPOINTER SETUP ---
    # In-memory checkpointer: same logic as a persistent store, but state clears
    # when the process stops. Swap for a durable saver in production.
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
