from langgraph.graph import StateGraph, START, END
from state import LogAgentState
from nodes import fetch_ssh_logs

def build_log_graph():

    builder = StateGraph(LogAgentState)
    
    builder.add_node("log_fetcher", fetch_ssh_logs)
    
    # START -> log_fetcher -> END
    builder.add_edge(START, "log_fetcher")
    builder.add_edge("log_fetcher", END)
    
    return builder.compile()