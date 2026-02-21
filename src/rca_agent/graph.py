from langgraph.graph import StateGraph, START, END
from .state import RCAState
from .nodes import reverse_search, differ_error, ollama_solver

def build_rca_graph():

    builder=StateGraph(RCAState)

    builder.add_node("reverse_search" , reverse_search)
    builder.add_node("differ_error" , differ_error)
    builder.add_node("ollama_solver" , ollama_solver)

    builder.set_entry_point("reverse_search")
    builder.add_edge("reverse_search" , "differ_error")
    builder.add_edge("differ_error" , "ollama_solver")
    builder.add_edge("ollama_solver" , END)

    return builder.compile()
