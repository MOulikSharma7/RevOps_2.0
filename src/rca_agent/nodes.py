import os
from langchain_ollama import OllamaLLM
from .state import RCAState

def reverse_search(state: RCAState):
    """Locates the latest failure in the log context."""
    print("--- Fetching Log Data...... ---")
    file_path = state.get('log_file_path', 'rca_agent/app.log')
    last_lines = ""

    if not os.path.exists(file_path):
        print(f"warning: Given file path {file_path} does not exist")
        return {"log_error": "No Log file found"}

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Get last 20 lines for context
            last_20 = lines[-20:] if len(lines) > 20 else lines
            last_lines = "".join(last_20)
    except Exception as e: 
        last_lines = f"Error reading log file: {str(e)}"

    return {"log_error": last_lines}

def differ_error(state: RCAState):
    """Uses Ollama to find the type of error by reading the log and exception."""
    print("--- Differentiating Error Type ---")
    
    llm = OllamaLLM(model="llama3.2-vision")
    
    # Combine context for the AI to analyze
    context = f"Exception: {state.get('exception_error', '')}\nLogs: {state.get('log_error', '')}"

    prompt = f"""
    Read the following logs and exception. 
    Identify what type of error this is (e.g., Syntax Error, Connection Issue, Dependency Missing, etc.).
    Respond with ONLY the error type name (maximum 3-5 words).

    ERROR DATA:
    {context}
    """

    # LLM decides the error type dynamically
    etype = llm.invoke(prompt).strip()

    print("="*50)
    print(f"DEBUG:Identified Category -> {etype.upper()}")
    print("="*50)
    return {"error_type": etype}

def ollama_solver(state: RCAState):
    """Generic solver that uses all state data to provide a technical fix."""
    print(f"--- Identifying Root Cause ---")

    llm = OllamaLLM(model="llama3.2-vision")

    # Generic Expert Persona as requested
    persona = "Persona: Expert Full-Stack Engineer and system Architect."

    prompt = f"""
    {persona}

    you are automated Root Cause Analysis (RCA) tool
    Analyze the following details to provide a detailed Technical Root Cause For the Following Issue:

    IDENTIFIED ERROR CATEGORY: {state.get('error_type')}
    EXCEPTION RECEIVED: {state.get('exception_error')}
    LOG CONTEXT: {state.get('log_error')}

    TASK:
    1. ROOT CAUSE: Explain the Root Cause based on the provided logs and exception.
    """

    response = llm.invoke(prompt)
    return {"root_cause": response}
