import os
from langchain_ollama import OllamaLLM
from .state import NbaState
# DeepEval integration is not yet wired up; see evaluate_nba below.
# from genesis.agents.deepeval_agent.graph import deepeval_app

def nba_solver(state: NbaState):
    """
    Generic NBA Specialist: Adaptable to any technical diagnosis.
    """

    print("--- NBA Agent: Generating Actionable Solution ---")

    # llama3.2 (non-vision) is highly recommended for stable text formatting
    #llm = OllamaLLM(model="llama3.2-vision")
    llm = OllamaLLM(model="qwen2.5-coder")

    inputs = state.get('inputs', [])
    rca_diagnosis = state.get('root_cause', 'No diagnosis provided')

    prompt = f"""
    SYSTEM ROLE: You are a expert Field Ops Engineer.
    You provide precise, actionable remediation steps for any system failure.

    [DIAGNOSTIC CONTEXT]
    Symptoms: {inputs}
    Confirmed Root Cause: {rca_diagnosis}

    TASK:
    Generate a technical "Next Best Action" plan to resolve the issue described above.

    STRICT OPERATIONAL RULES:
    - Provide only the direct solution steps.
    - Do not include an introduction, a summary, or categories.
    - List the steps in chronological order (e.g., Step 1, Step 2).
    - Ensure each step is actionable and specific to the provided RCA.
    - If a specific command or code fix is required, provide it clearly
    - Generate a set of Remediation Steps to resolve the issue"""

    response = llm.invoke(prompt)

    return {"suggested_fix": response}

def evaluate_nba(state: NbaState):
    """
    NEW NODE: Integrates with DeepEval Agent.
    Validates the NBA solution against the symptoms and RCA diagnosis.

    NOTE: Not yet wired up -- the DeepEval agent (deepeval_app) is unavailable,
    so this node is currently excluded from the graph and raises if invoked.
    """

    print("--- Evaluating NBA Output via DeepEval ---")

    raise NotImplementedError(
        "evaluate_nba requires the DeepEval agent, which is not yet integrated. "
        "Wire up deepeval_app before enabling this node in the graph."
    )
