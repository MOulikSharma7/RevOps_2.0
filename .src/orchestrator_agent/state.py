System Persona: Act as an expert Developer Advocate and LinkedIn Copywriter. I am providing you with the source code (or a detailed summary) of my latest software project below. Please analyze it and write a highly engaging LinkedIn post to showcase it.
Follow this exact structure and set of rules:

The Hook (First 2-3 lines): Create a scroll-stopping opening that highlights the core problem solved or the technical "aha!" moment. Do NOT use generic openings like "I am excited to share" or "I've been working on." Make it punchy and bold.
The Context/Problem: Explain why this tool needed to exist in 1-2 short sentences. What specific pain point or bottleneck does it solve?
The Solution & Architecture: Explain what the code does in plain English. Explicitly highlight the tech stack (extract the specific programming languages, libraries, machine learning models, and database architectures from the code).
Key Takeaways: Provide a 3-bullet list detailing the most interesting technical features, performance improvements, or logic hurdles overcome during development. Focus on the engineering depth.
Formatting: Use short, single-sentence paragraphs. Leave plenty of whitespace so it is easy to read on mobile. Use emojis strategically to break up text, but keep it highly professional.
Call to Action (CTA): End with an open-ended question to spark discussion in the comments, and tell readers the GitHub repository link is available below.
Hashtags: Include 4-5 targeted hashtags strictly relevant to the tech stack and industry.

-Emphasize Agentic AI and Gen AI  - Langgraph framework
- Emphasize how I implemented the multi-agent workflow (dynamic supervisor agent with similarity search for agent retrieval , vector db similarity search) 
-  MongoDb Mcp server in db agent (although i haven't done it but assume im using it )



Here is the project code:
Orchestrator agent -
genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/Orchestrator_Agent$ cat nodes.py
# /home/genaidevassetv3/FieldOps_2.0/Orchestrator_Agent/nodes.py

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

genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/Orchestrator_Agent$ cat prompts.py
# /home/genaidevassetv3/FieldOps_2.0/Orchestrator_Agent/prompts.py

AVAILABLE_AGENTS = {
    "db_agent": (
        "Smart Database Query Engine. "
        "Capabilities: Translates natural language requests into executable MongoDB queries. "
        "Can retrieve ANY infrastructure data including Server IPs, OS details, Active Alarms, Site locations, and historical records. 
"
        "Input: A clear natural language description of the data you need (e.g., 'Find the IP for beta-02' or 'List critical alarms')."
    ),
    "log_retriever_agent": (
        "Log File Retriever. "
        "Requirements: Needs a valid 'Host/IP' and 'User' and 'Password' to connect. "
        "Capabilities: Connects to the server via SSH to fetch system logs (journalctl). "
        "Smart: Automatically retrieves IP and Credentials from the Database if not provided. "
        "Provides: The file path of the saved logs."
    ),
    "RCA_agent": (
        "Root Cause Analyzer. "
        "Requirements: Needs 'Raw Logs' and 'Alarm Context' to perform analysis. "
        "Provides: A technical diagnosis of the root cause of failure."
    ),
    "NBA_agent": (
        "Next Best Action Recommender. "
        "Requirements: Needs a 'Root Cause Analysis Report'. "
        "Provides: A recommended set of remediation steps."
    ),
    "automation_agent": (
        "Remediation Scripter. "
        "Requirements: Needs 'Recommended Actions' and 'Server Details'. "
        "Provides: An executable script (Python/Bash) to fix the issue."
    ),
    "execute_agent": (
        "Command Executor. "
        "Requirements: Needs a 'Remediation Script' and 'Valid Credentials'. "
        "Provides: Execution results (Success/Failure)."
    ),
    "RAG_agent": (
        "Knowledge Base. "
        "Capabilities: Searches technical documentation. "
        "Provides: Answers from SOPs, Manuals, and guides."
    )
}

SUPERVISOR_SYSTEM_PROMPT = """
You are a FieldOps Supervisor. Your goal is to complete the user's request by coordinating the available sub-agents.

Available Agents:
{agent_list}

Instructions:
1. Analyze the User's Request.
2. Identify the **Data Dependencies**:
   - Check if you have the necessary inputs for the agent you want to call.
   - If inputs are missing (e.g., missing IP address), call the agent that provides them.
   - **Database Information Needs**: Act as an expert FieldOps Engineer to determine exactly what information is required. Provide a cl
ear dialogue describing the specific details you need from the database. This description will allow the db_agent to handle the technic
al query generation.
3. **THINKING PROCESS**: Briefly explain your reasoning.
4. **FINAL COMMAND**: You MUST end your response with a specific line stating the action.

FORMATTING RULE:
Your final line must ALWAYS look like this:
ACTION: [agent_name]

Example:
"I need the device's current status and location to begin the diagnostic. 
Information Needed: I need to find the latest telemetry and site location for the device with ID 'DEV-99'.
ACTION: db_agent"

If the task is complete:
"I have finished the task.
ACTION: FINISH"

CRITICAL RULES:
- **TRUST YOUR MEMORY**: If the output of a tool (like IP address or Logs) is already in the conversation history, DO NOT call that too
l again. Use the existing data.
- **CHECK HISTORY**: If a tool has already run, DO NOT call it again for the same purpose.
- **SCOPE RESTRICTION**: Do NOT voluntarily perform steps that were not requested. 
    - If the user asks for "Analysis" or "RCA", STOP after the RCA_agent runs. Do NOT proceed to NBA or Automation.
    - If the user asks for "Remediation" or "Fix", ONLY THEN proceed to NBA and Automation.
- **OUTPUT**: Ensure the very last line is "ACTION: [agent_name]".
"""

genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/Orchestrator_Agent$ cat graph.py
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

db agent -
genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/SubAgents/db_agent$ cat nodes.py
import json
import os
import ollama
from pymongo import MongoClient
from bson import json_util, ObjectId
from state import AgentState
from prompts import TRANSFORM_PROMPT, GENERATE_QUERY_PROMPT

# --- LOAD VARIABLES FROM CONFIG ---
DB_CONNECTION = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_NAME", "FieldOps")

if not DB_NAME:
    raise ValueError("MONGODB_NAME is missing! Please set it in your environment.")

MODEL_NAME = "qwen2.5-coder:14b"
CACHE_FILE = "db_schema_cache.json"

class DBAgentNodes:
    def __init__(self):
        try:
            print(f"Connecting to Database: '{DB_NAME}' at '{DB_CONNECTION}'...")
            self.client = MongoClient(DB_CONNECTION)
            self.db = self.client[DB_NAME]
            
            # Quick check to verify connection
            self.client.admin.command('ping')
            print(f"Connected to MongoDB: {DB_NAME}")
        except Exception as e:
            print(f"Connection Failed: {e}")
            self.db = None

    def _get_schema(self, force_refresh=False):
        if not force_refresh and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)["schema_text"]
            except Exception:
                pass
        return self._scan_and_cache_db()

    def _scan_and_cache_db(self):
        if self.db is None:
            return "Error: No database connection."
            
        print("Scanning database structure...")
        schema_text = ""
        try:
            for col_name in self.db.list_collection_names():
                if "system." in col_name:
                    continue
                sample = self.db[col_name].find_one()
                if sample:
                    fields = [f"{k} ({type(v).__name__})" for k, v in sample.items() if k != "_id"]
                    schema_text += f"- Collection: '{col_name}' | Fields: {', '.join(fields)}\n"
            
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"schema_text": schema_text}, f)
        except Exception as e:
            print(f"Schema scan failed: {e}")
            
        return schema_text

    # --- NODE 1: LOAD SCHEMA ---
    def load_schema_node(self, state: AgentState) -> dict:
        # FIX: Force refresh is now True. It will always scan the live DB.
        schema = self._get_schema(force_refresh=True)
        return {"schema_context": schema}

    # --- NODE 2: TRANSFORM QUERY ---
    def transform_query_node(self, state: AgentState) -> dict:
        print(f"Transforming: '{state['raw_input']}'")
        try:
            prompt = TRANSFORM_PROMPT.format(schema_context=state['schema_context'])
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": state['raw_input']}
                ]
            )
            return {"clean_question": response["message"]["content"].strip()}
        except Exception as e:
            return {"error": f"Transformation Failed: {e}"}

    # --- NODE 3: GENERATE JSON ---
    def generate_json_node(self, state: AgentState) -> dict:
        print(f"Generating JSON for: '{state['clean_question']}'")
        try:
            prompt = GENERATE_QUERY_PROMPT.format(
                schema_context=state['schema_context'],
                clean_question=state['clean_question']
            )
            response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            
            clean_json = response["message"]["content"].replace("```json", "").replace("```", "").strip()
            start, end = clean_json.find("{"), clean_json.rfind("}") + 1
            if start != -1 and end != -1:
                clean_json = clean_json[start:end]
            
            return {"query_json": json.loads(clean_json)}
        except Exception as e:
            return {"error": f"JSON Generation Failed: {e}"}

    # --- NODE 4: EXECUTE & FORMAT ---
    def execute_query_node(self, state: AgentState) -> dict:
        q_data = state.get('query_json')
        if not q_data:
            return {"error": "No query JSON available.", "final_response": "Failed to generate query."}
            
        target_col = q_data.get("collection")
        query_filter = q_data.get("query")
        
        print(f"Target: {target_col} | Filter: {query_filter}")
        
        try:
            # Fetch results
            cursor = self.db[target_col].find(query_filter).limit(5)
            results = list(cursor)
            
            # --- SANITIZE OBJECTID ---
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            # -------------------------

            # Format results for the Supervisor
            if results:
                result_str = json.dumps(results, indent=2, default=str)
                summary = f"Successfully retrieved {len(results)} records from '{target_col}'."
            else:
                result_str = "[]"
                summary = f"No records found in '{target_col}' matching criteria."

            return {
                "query_result": results,
                "final_response": summary + "\nData: " + result_str
            }
        except Exception as e:
            return {"error": str(e), "final_response": f"Database Execution Error: {str(e)}"}

genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/SubAgents/db_agent$ cat graph.py
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
genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/SubAgents/db_agent$ cat prompts.py
# /home/genaidevassetv3/FieldOps_2.0/SubAgents/db_agent/prompts.py

# --- SYSTEM PROMPT (Transformer) ---
TRANSFORM_PROMPT = """
You are a Database Expert and MongoDB Engineer. Your role is to translate natural language requests from the FieldOps Supervisor into e
fficient MongoDB queries.

Current Database Schema:
{schema_context}

Instructions:
1. **Interpret Needs**: Carefully read the 'Information Needed' dialogue from the Supervisor.
2. **Schema Mapping**: Identify which collection(s) contain the required data based ONLY on the provided schema. Do not guess collectio
ns that are not listed.
3. **Refinement**: Translate the complex or vague request into a clear, single-sentence question that describes exactly what data to fe
tch.

CRITICAL OUTPUT RULE:
- Do NOT output code or JSON here.
- Do NOT hallucinate data or collections.
- Output ONLY the refined natural language question.
"""

# --- GENERATION PROMPT (JSON Builder) ---
GENERATE_QUERY_PROMPT = """
You are a MongoDB Expert. Given the database schema below, write a MongoDB Query Object (JSON) to answer the user's question.

{schema_context}

User Question: "{clean_question}"

**CRITICAL RULES FOR QUERY GENERATION:**
1. **Output ONLY a valid JSON object**. No comments, no code blocks, no explanation text.
2. **Format**: {{ "collection": "TargetCollectionName", "query": {{ ...MQL_FILTER... }} }}
3. **Collection Mapping**: ONLY use a collection name that exactly matches one listed in the schema above. Do NOT use old names like 'A
larms' or 'Customers' unless they appear in the schema.
4. **Synonym Handling (CRITICAL)**: 
   - Humans use imprecise language. You must map their words to the EXACT values and fields in the schema.
   - Example: If a user asks for "active faults", look for a collection like 'faults' and a status field like 'Status' with a value of 
'Active'.
5. **Value Matching (CRITICAL UPDATE)**:
   - **ALWAYS use Regex** for descriptive fields (Names, OS, Descriptions, Types, IDs) to capture partial matches.
   - **Example**: Instead of {{ "Server_Name": "genaidevassetv3" }}, use {{ "Server_Name": {{ "$regex": "genaidevassetv3", "$options": 
"i" }} }}.

Example Output Format:
{{ "collection": "ActualCollectionNameFromSchema", "query": {{ "FieldFromSchema": {{ "$regex": "SearchTerm", "$options": "i" }} }} }}
"""
rca agent-
genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/SubAgents/rca_agent$ cat graph.py
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
genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/SubAgents/rca_agent$ cat nodes.py
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
nba agent -
genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/SubAgents/nba_agent$ cat graph.py 
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
genaidevassetv3@genaidevassetv3:~/FieldOps_2.0/SubAgents/nba_agent$ cat nodes.py
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

and log retriever agent .