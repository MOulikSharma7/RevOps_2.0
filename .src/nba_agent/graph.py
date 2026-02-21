from langgraph.graph import StateGraph, END , START
from state import NbaState
from nodes import nba_solver, evaluate_nba

def build_nba_graph():

    workflow = StateGraph(NbaState)

    workflow.add_node("nba_solver", nba_solver)
#    workflow.add_node("evaluate_nba", evaluate_nba)
    
    workflow.set_entry_point("nba_solver")
#    workflow.add_edge("nba_solver", "evaluate_nba")
#    workflow.add_edge("evaluate_nba", END)
    workflow.add_edge("nba_solver", END)
    
    return workflow.compile()
