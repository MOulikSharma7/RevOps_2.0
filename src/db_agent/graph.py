from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import DBAgentNodes

def create_graph():
    nodes = DBAgentNodes()
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("load_schema", nodes.load_schema_node)
    workflow.add_node("transform", nodes.transform_query_node)
    workflow.add_node("generate_json", nodes.generate_json_node)
    workflow.add_node("execute_db", nodes.execute_query_node)
    
    # Define Flow
    workflow.set_entry_point("load_schema")
    workflow.add_edge("load_schema", "transform")
    workflow.add_edge("transform", "generate_json")
    workflow.add_edge("generate_json", "execute_db")
    workflow.add_edge("execute_db", END)
    
    return workflow.compile()
