# /home/genaidevassetv3/FieldOps_2.0/Orchestrator_Agent/graph.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # <--- CHANGED: Use MemorySaver
from Orchestrator_Agent.state import IntentState
from Orchestrator_Agent import nodes

def route_decision(state: IntentState) -> str:
    decision = state["selected_agent"]
    
    if decision == "FINISH":
        print("DEBUG: Routing to END")
        return END
        
    mapping = {
        "db_agent": "db_node",
        "log_retriever_agent": "log_node",
        "RCA_agent": "rca_node",
        "NBA_agent": "nba_node",
        "automation_agent": "automation_node",
        "execute_agent": "execute_node",
        "rag_agent": "rag_node"
    }
    
    next_node = mapping.get(decision, END)
    
    if next_node == END and decision != "FINISH":
        print(f"WARNING: Could not map decision '{decision}' to a node. Defaulting to END.")
    else:
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
            END: END
        }
    )
    
    agents = ["db_node", "log_node", "rca_node", "nba_node", "automation_node", "execute_node", "rag_node"]
    for agent_node in agents:
        workflow.add_edge(agent_node, "supervisor_node")

    # --- MEMORY CHECKPOINTER SETUP ---
    # We use the simple in-memory checkpointer. 
    # It works exactly like Redis for logic, but data clears when script stops.
    checkpointer = MemorySaver()

    # Compile with the checkpointer
    return workflow.compile(checkpointer=checkpointer)

