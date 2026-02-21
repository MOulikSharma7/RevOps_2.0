import sys
import os
import re
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Define paths to SubAgents
db_agent_path = os.path.join(project_root, 'SubAgents', 'db_agent')
log_agent_path = os.path.join(project_root, 'SubAgents', 'log_agent')
rca_agent_path = os.path.join(project_root, 'SubAgents', 'rca_agent')
nba_agent_path = os.path.join(project_root, 'SubAgents', 'nba_agent')
automation_agent_path = os.path.join(project_root, 'SubAgents', 'automation_agent')

from llm.LlamaIndexManager import clsLlamaIndexManager
from Orchestrator_Agent.prompts import AVAILABLE_AGENTS, SUPERVISOR_SYSTEM_PROMPT
from Orchestrator_Agent.state import IntentState
from SubAgents import agents as dummy_agents 

# --- REAL AGENT IMPORTS (With Cache Clearing) ---

def clear_module_cache():
    """Clears SubAgent module names from sys.modules to prevent naming collisions."""
    for module_name in ['state', 'nodes', 'graph']:
        if module_name in sys.modules:
            del sys.modules[module_name]

# 1. Import DB Agent
try:
    sys.path.insert(0, db_agent_path)
    clear_module_cache()
    from SubAgents.db_agent.graph import create_graph as create_db_graph
    print("[INFO] Successfully imported Real DB Agent.")
except ImportError as e:
    print(f"[ERROR] Could not import DB Agent: {e}")
    create_db_graph = None
finally:
    if db_agent_path in sys.path: sys.path.remove(db_agent_path)

# 2. Import Log Agent
try:
    sys.path.insert(0, log_agent_path)
    clear_module_cache()
    from SubAgents.log_agent.graph import build_log_graph
    print("[INFO] Successfully imported Real Log Agent.")
except ImportError as e:
    print(f"[ERROR] Could not import Log Agent: {e}")
    build_log_graph = None
finally:
    if log_agent_path in sys.path: sys.path.remove(log_agent_path)

# 3. Import RCA Agent
try:
    sys.path.insert(0, rca_agent_path)
    clear_module_cache()
    from SubAgents.rca_agent.graph import build_rca_graph
    print("[INFO] Successfully imported Real RCA Agent.")
except ImportError as e:
    print(f"[ERROR] Could not import RCA Agent: {e}")
    build_rca_graph = None
finally:
    if rca_agent_path in sys.path: sys.path.remove(rca_agent_path)

# 4. Import NBA Agent
try:
    sys.path.insert(0, nba_agent_path)
    clear_module_cache()
    from SubAgents.nba_agent.graph import build_nba_graph
    print("[INFO] Successfully imported Real NBA Agent.")
except ImportError as e:
    print(f"[ERROR] Could not import NBA Agent: {e}")
    build_nba_graph = None
finally:
    if nba_agent_path in sys.path: sys.path.remove(nba_agent_path)

# 5. Import Automation Agent
try:
    sys.path.insert(0, automation_agent_path)
    clear_module_cache()
    from SubAgents.automation_agent.graph import build_automation_graph
    print("[INFO] Successfully imported Real Automation Agent.")
except ImportError as e:
    print(f"[ERROR] Could not import Automation Agent: {e}")
    build_automation_graph = None
finally:
    if automation_agent_path in sys.path: sys.path.remove(automation_agent_path)


# --- Initialization ---
llm_manager = clsLlamaIndexManager.get_instance()
_AGENT_INDEX = None

# Initialize Agents
db_agent_app = create_db_graph() if create_db_graph else None
log_agent_app = build_log_graph() if build_log_graph else None
rca_agent_app = build_rca_graph() if build_rca_graph else None
nba_agent_app = build_nba_graph() if build_nba_graph else None
automation_agent_app = build_automation_graph() if build_automation_graph else None

def get_agent_index():
    global _AGENT_INDEX
    if _AGENT_INDEX: return _AGENT_INDEX
    documents = [Document(text=desc, metadata={"agent_name": name}) for name, desc in AVAILABLE_AGENTS.items()]
    _AGENT_INDEX = VectorStoreIndex.from_documents(documents)
    return _AGENT_INDEX

def get_retriever(top_k=5):
    return get_agent_index().as_retriever(similarity_top_k=top_k)


# --- Core Logic Nodes ---

def retrieve_agents_node(state: IntentState):
    print("--- Node: Retrieving Relevant Agents ---")
    initial_query = state["messages"][0]["content"]
    
    retriever_narrow = get_retriever(top_k=4)
    nodes = retriever_narrow.retrieve(initial_query)
    
    trigger_fallback = not nodes or nodes[0].score < 0.74
    if trigger_fallback:
        print("⚠ Low Confidence in retrieval. Expanding search window (Fallback).")
        nodes = get_retriever(top_k=10).retrieve(initial_query)
    
    retrieved_names = [n.metadata["agent_name"] for n in nodes]
    descriptions = "\n".join([f"- {n.metadata['agent_name']}: {n.text}" for n in nodes])
    
    return {"possible_agents": retrieved_names, "agent_descriptions": descriptions}

def supervisor_node(state: IntentState):
    print("--- Node: Supervisor Thinking ---")
    
    desc = state["agent_descriptions"]
    system_msg = ChatMessage(role=MessageRole.SYSTEM, content=SUPERVISOR_SYSTEM_PROMPT.format(agent_list=desc))
    
    chat_history = [system_msg]
    for msg in state["messages"]:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        chat_history.append(ChatMessage(role=role, content=msg["content"]))
    
    response = Settings.llm.chat(chat_history)
    raw_decision = response.message.content.strip()
    
    print(f"Supervisor Raw Output Snippet:\n'{raw_decision[:100]}...'")

    match = re.search(r"ACTION:\s*(\w+)", raw_decision, re.IGNORECASE)

    if match:
        decision = match.group(1).strip()
    else:
        valid_agents = list(AVAILABLE_AGENTS.keys()) + ["FINISH"]
        all_matches = re.findall(r"\b(" + "|".join(valid_agents) + r")\b", raw_decision)
        if all_matches:
            decision = all_matches[-1]
            print(f"WARNING: Fallback to last mention: {decision}")
        else:
            decision = "FINISH"
            print("ERROR: Could not parse agent. Defaulting to FINISH.")

    # --- HARD SCOPE GUARDRAIL ---
    if state["messages"]:
        initial_query = state["messages"][0]["content"].lower()
        last_message = state["messages"][-1]["content"]

        # Guardrail 1: Stop after RCA if NBA was not requested
        if "OUTPUT from RCA_agent" in last_message:
            if "next best action" not in initial_query and "automate" not in initial_query:
                print(f"GUARDRAIL ACTIVE: RCA is done. Forcing FINISH to prevent auto-triggering {decision}.")
                decision = "FINISH"

        # Guardrail 2: Stop after NBA if Automation was not requested
        elif "OUTPUT from NBA_agent" in last_message:
            if "automate" not in initial_query and "script" not in initial_query:
                print(f"GUARDRAIL ACTIVE: NBA is done. Forcing FINISH to prevent auto-triggering {decision}.")
                decision = "FINISH"
                
        # Guardrail 3: Stop after Automation (prevents accidental execution)
        elif "OUTPUT from automation_agent" in last_message:
            if "execute" not in initial_query:
                print(f"GUARDRAIL ACTIVE: Script generated. Forcing FINISH to prevent auto-triggering {decision}.")
                decision = "FINISH"
    # ----------------------------

    if state["messages"] and f"OUTPUT from {decision}" in state["messages"][-1]["content"]:
        print(f"WARNING: Supervisor selected '{decision}' again immediately after it ran. Forcing FINISH.")
        decision = "FINISH"

    print(f"Supervisor Final Decision: '{decision}'")
    return {"selected_agent": decision, "messages": [{"role": "assistant", "content": raw_decision}]}


# --- Execution Nodes ---

def _run_agent_generic(state: IntentState, agent_func, agent_name):
    output = agent_func(state["messages"][-1]["content"])
    return {"messages": [{"role": "user", "content": f"OUTPUT from {agent_name}:\n{output}"}]}


# --- REAL DB AGENT ---
def run_db_agent(state: IntentState):
    print("--- Executing REAL DB AGENT (Sub-Graph) ---")
    if not db_agent_app: return {"messages": [{"role": "user", "content": "OUTPUT from db_agent:\n[ERROR] DB Agent failed to load."}]}
    try:
        result = db_agent_app.invoke({"raw_input": state["messages"][-1]["content"]})
        output_text = result.get("final_response") or (f"DB Error: {result['error']}" if result.get("error") else "No response.")
    except Exception as e:
        output_text = f"CRITICAL ERROR in DB Agent Execution: {str(e)}"
    return {"messages": [{"role": "user", "content": f"OUTPUT from db_agent:\n{output_text}"}]}


# --- REAL LOG RETRIEVER AGENT ---
def run_log_retriever_agent(state: IntentState):
    print("--- Executing REAL LOG AGENT (Sub-Graph) ---")
    if not log_agent_app: return {"messages": [{"role": "user", "content": "OUTPUT from log_retriever_agent:\n[ERROR] Log Agent failed 
to load."}]}

    history_text = "\n".join([m["content"] for m in state["messages"]])
    
    # 1. STRICT Hostname Extraction (No Defaults!)
    target_host = None
    
    # Regex matches: 'genaidevassetv1', 'VM-01', 'beta-02', 'server.domain.com', '10.1.0.5'
    match = re.search(r'\b(genaidevassetv\d+|VM-\d+|beta-\d{2}|[a-zA-Z0-9-]+\.\d+|10\.1\.\d+\.\d+)\b', history_text, re.IGNORECASE)
    
    if match: 
        target_host = match.group(0)
    else:
        # HARD STOP: If we can't find a host, we can't fetch logs.
        print("[ERROR] No valid hostname found in request.")
        return {"messages": [{"role": "user", "content": "OUTPUT from log_retriever_agent:\n[ERROR] Could not identify a valid Hostname
, Node ID, or IP in the request. Please provide the server name (e.g., genaidevassetv1) or IP."}]}
    
    print(f"DEBUG: Identified Target Host: '{target_host}'. Querying DB for credentials...")
    
    # 2. Query DB for Credentials
    db_inputs = {"raw_input": f"Get the IP_Address, User, and Password for the server '{target_host}'"}
    creds = {}
    
    try:
        db_result = db_agent_app.invoke(db_inputs)
        raw_data = db_result.get("query_result", [])
        
        # Handle list of dicts from DB
        if raw_data and isinstance(raw_data, list) and len(raw_data) > 0:
            creds["host"] = raw_data[0].get("IP_Address")
            creds["user"] = raw_data[0].get("User")
            creds["password"] = raw_data[0].get("Password")
        # Handle direct dict return
        elif raw_data and isinstance(raw_data, dict):
            creds["host"] = raw_data.get("IP_Address")
            creds["user"] = raw_data.get("User")
            creds["password"] = raw_data.get("Password")
            
    except Exception as e: print(f"ERROR: Failed to fetch DB credentials: {e}")

    # 3. Validation: Did we get credentials?
    missing_fields = [k for k in ["host", "user", "password"] if not creds.get(k)]
    if missing_fields:
        return {"messages": [{"role": "user", "content": f"OUTPUT from log_retriever_agent:\n[ERROR] Failed to retrieve credentials for
 '{target_host}'. Missing DB fields: {', '.join(missing_fields)}"}]}

    # 4. Fetch Logs
    try:
        result = log_agent_app.invoke(creds)
        status = result.get("status", "unknown")
        if "success" in status:
            output_text = f"Successfully retrieved system logs from {target_host} ({creds['host']}).\nLogs saved to: {result.get('log_f
ile_path')}\nStatus: {status}"
        else:
            output_text = f"Failed to retrieve logs. Error: {status}"
    except Exception as e:
        output_text = f"CRITICAL ERROR in Log Agent: {str(e)}"

    return {"messages": [{"role": "user", "content": f"OUTPUT from log_retriever_agent:\n{output_text}"}]}


# --- REAL RCA AGENT ---
def run_rca_agent(state: IntentState):
    print("--- Executing REAL RCA AGENT (Sub-Graph) ---")
    if not rca_agent_app: return {"messages": [{"role": "user", "content": "OUTPUT from RCA_agent:\n[ERROR] RCA Agent failed to load."}
]}

    history_text = "\n".join([m["content"] for m in state["messages"]])
    log_file_path = None
    
    # Matches /tmp paths
    match = re.search(r'(/tmp/[a-zA-Z0-9._-]+)', history_text)
    if match: log_file_path = match.group(1)

    if not log_file_path:
        return {"messages": [{"role": "user", "content": f"OUTPUT from RCA_agent:\n[ERROR] No log file path found in history. Please ru
n Log Retriever first."}]}

    print(f"DEBUG: Found Log File Path for RCA: {log_file_path}")
    try:
        # Pass context from query
        initial_query = state["messages"][0]["content"]
        result = rca_agent_app.invoke({"log_file_path": log_file_path, "exception_error": initial_query})
        
        output_text = (
            f"Root Cause Analysis Complete.\n\n"
            f"Identified Error Type: {result.get('error_type', 'Unknown')}\n"
            f"Technical Diagnosis:\n{result.get('root_cause', 'No root cause identified.')}"
        )
    except Exception as e:
        output_text = f"CRITICAL ERROR in RCA Agent: {str(e)}"

    return {"messages": [{"role": "user", "content": f"OUTPUT from RCA_agent:\n{output_text}"}]}


# --- REAL NBA AGENT ---
def run_nba_agent(state: IntentState):
    print("--- Executing REAL NBA AGENT (Sub-Graph) ---")
    if not nba_agent_app: 
        return {"messages": [{"role": "user", "content": "OUTPUT from NBA_agent:\n[ERROR] NBA Agent failed to load."}]}

    rca_diagnosis = "No prior diagnosis found."
    for msg in reversed(state["messages"]):
        if msg["role"] == "user" and "OUTPUT from RCA_agent:" in msg["content"]:
            rca_diagnosis = msg["content"].replace("OUTPUT from RCA_agent:\n", "").strip()
            break
            
    initial_query = state["messages"][0]["content"] if state["messages"] else "Unknown Issue"

    print("DEBUG: Extracted RCA context and User Query for NBA.")

    output_text = ""
    try:
        inputs = {
            "inputs": [initial_query],
            "root_cause": rca_diagnosis
        }
        
        result = nba_agent_app.invoke(inputs)
        suggested_fix = result.get("suggested_fix", "No fix generated.")
        
        output_text = (
            f"Next Best Action (Remediation Plan) Generated:\n\n"
            f"{suggested_fix}"
        )
        
    except Exception as e:
        output_text = f"CRITICAL ERROR in NBA Agent Execution: {str(e)}"
        print(output_text)

    return {"messages": [{"role": "user", "content": f"OUTPUT from NBA_agent:\n{output_text}"}]}


# --- REAL AUTOMATION AGENT ---
def run_automation_agent(state: IntentState):
    print("--- Executing REAL AUTOMATION AGENT (Sub-Graph) ---")
    if not automation_agent_app: 
        return {"messages": [{"role": "user", "content": "OUTPUT from automation_agent:\n[ERROR] Automation Agent failed to load."}]}

    # 1. Extract the NBA Plan from the Chat History
    nba_plan = "No previous NBA plan found."
    for msg in reversed(state["messages"]):
        if msg["role"] == "user" and "OUTPUT from NBA_agent:" in msg["content"]:
            nba_plan = msg["content"].replace("OUTPUT from NBA_agent:\n", "").strip()
            break

    print("DEBUG: Extracted NBA Plan for Automation Scripting.")

    # 2. Invoke Automation Sub-Graph
    output_text = ""
    try:
        inputs = {
            "nba_plan": nba_plan,
            "target_os": "Linux" # We can make this dynamic later
        }
        
        result = automation_agent_app.invoke(inputs)
        script_content = result.get("script_content", "# No script generated")
        script_type = result.get("script_type", "bash")
        
        # Format the output for the Supervisor
        output_text = (
            f"Automation Script Generated ({script_type}):\n\n"
            f"```bash\n{script_content}\n```\n\n"
            f"The script is ready for execution."
        )
        
    except Exception as e:
        output_text = f"CRITICAL ERROR in Automation Agent Execution: {str(e)}"
        print(output_text)

    return {"messages": [{"role": "user", "content": f"OUTPUT from automation_agent:\n{output_text}"}]}


# Wrapper for the remaining Dummy execute_agent
def run_execute_agent(state: IntentState): return _run_agent_generic(state, dummy_agents.execute_agent, "execute_agent")
def run_rag_agent(state: IntentState): return _run_agent_generic(state, dummy_agents.RAG_agent, "RAG_agent")

