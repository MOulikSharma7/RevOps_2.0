from typing import Optional, List, Any, TypedDict

class AgentState(TypedDict):
    """
    Shared state for the DB Agent.
    The 'final_response' is what the Supervisor Agent will consume.
    """
    # Inputs
    raw_input: str
    schema_context: str

    # Internal Processing
    clean_question: Optional[str]
    query_json: Optional[dict]

    # Outputs
    query_result: Optional[List[Any]]   # Raw MongoDB documents (List of Dicts)
    final_response: Optional[str]       # Natural language summary for Supervisor
    error: Optional[str]                # Error message if any

        