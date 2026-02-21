import os
from langchain_ollama import OllamaLLM
from state import NbaState
#from genesis.agents.deepeval_agent.graph import deepeval_app

def nba_solver(state: NbaState):
    """
    Generic NBA Specialist: Adaptable to any technical diagnosis.
    """

    print("--- NBA Agent: Generating Actionable Solution ---")

    # llama3.2 (non-vision) is highly recommended for stable text formatting
    #llm = OllamaLLM(model="llama3.2-vision") 
    llm = OllamaLLM(model = "qwen2.5-coder")
    
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
    - Generate a set of Renediation Steps to resolve the issue"""
     
    response = llm.invoke(prompt)

    return {"suggested_fix" : response}

def evaluate_nba(state: NbaState):
    """ 
    NEW NODE: Integrates with DeepEval Agent. 
    Validates the NBA solution against the symptoms and RCA diagnosis. 
    """ 
    
    print("--- Evaluating NBA Output via DeepEval ---")
   
    eval_input = { 
        "user_input": f"Provide remediation steps for: {state.get('root_cause')}", 
        "actual_output": state.get("suggested_fix", ""), 
        "retrieval_context": [str(state.get("inputs","")), state.get("root_cause", "")], 
        "threshold": 0.5
    }
    
    # Invoke the DeepEval Graph
    eval_result = deepeval_app.invoke(eval_input)
    # Output the evaluation results to the console
    print(f"NBA Evaluation Score:{eval_result.get('score')}") 
    print(f"NBA Evaluation Success: {eval_result.get('is_successful')}") 
    print(f"NBA Reason: {eval_result.get('reason')}")
 
    return { 
        "evaluation_score": eval_result.get('score'), 
        "is_valid": eval_result.get('is_successful')
    }
