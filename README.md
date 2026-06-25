# 🛠️ RevOps 2.0: Autonomous Multi-Agent IT Remediation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-black)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**RevOps 2.0** is an autonomous, multi-agent AI framework for automating IT operations, root cause analysis (RCA), and incident remediation. Built on **LangGraph**, it coordinates specialized sub-agents that query databases, fetch server logs over SSH, diagnose errors, and propose actionable technical fixes — all using **local LLMs via Ollama**.

---

## 🚀 Architecture & Sub-Agents

An **Orchestrator Agent** drives the workflow. It uses a LlamaIndex vector retriever to shortlist relevant sub-agents for a request, then a Supervisor LLM picks the next action each turn. State is held in a LangGraph checkpointer, and scope guardrails prevent the supervisor from over-running (e.g. auto-triggering remediation when only analysis was requested).

| Agent | Role | Model / Tech |
|-------|------|--------------|
| 🧠 **Orchestrator** | Supervisor + semantic agent routing, state memory, guardrails | LlamaIndex retriever + Ollama LLM |
| 🗄️ **DB Agent** | Translates natural language into MongoDB queries, executes them, returns infra context | `qwen2.5-coder:14b`, `pymongo` |
| 📡 **Log Retriever** | Opens an SSH session to fetch live `journalctl` logs for a node | `paramiko` |
| 🔍 **RCA Agent** | Analyzes exceptions against retrieved logs to pinpoint the failure | `llama3.2-vision` |
| 💡 **NBA Agent** | Turns the RCA report into a chronological remediation plan | `qwen2.5-coder` |
| ⚙️ **Automation Agent** *(in development)* | Scaffolds NBA plans into Python/Bash scripts | deterministic stub |
| 🧩 **execute_agent / RAG_agent** *(stubs)* | Command execution and knowledge-base lookup | placeholder responses |

## 💻 Tech Stack

* **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) & LangChain
* **Local LLMs**: [Ollama](https://ollama.com/) (`qwen2.5-coder:14b`, `llama3.2-vision`)
* **Routing & Retrieval**: LlamaIndex `VectorStoreIndex` + Ollama embeddings
* **Database**: MongoDB (via `pymongo`)
* **Infrastructure**: `paramiko` (SSH automation)

## 📂 Project Structure

```
RevOps_2.0/
├── main.py                     # Interactive entry point (REPL against the orchestrator)
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Runtime + test dependencies
├── pytest.ini                  # Test configuration (adds src/ to import path)
├── .env.example                # Sample configuration
├── src/
│   ├── orchestrator_agent/     # Supervisor: retriever, supervisor, routing, graph
│   ├── db_agent/               # NL → MongoDB query agent
│   ├── log_retriever_agent/    # SSH log fetcher
│   ├── rca_agent/              # Root cause analysis
│   ├── nba_agent/              # Next-best-action planner
│   ├── automation_agent/       # Remediation script scaffolder (in development)
│   ├── agents/                 # execute_agent / RAG_agent placeholders
│   └── llm/                    # LlamaIndex + Ollama runtime configuration
└── tests/                      # Offline unit tests (no DB/Ollama required)
```

Each sub-agent is a self-contained LangGraph package (`state.py`, `nodes.py`, `graph.py`). Sub-graphs are imported and orchestrated by `src/orchestrator_agent/nodes.py`.

## 🛠️ Quickstart

### 1. Prerequisites
* Python 3.10+
* MongoDB running locally or remotely
* [Ollama](https://ollama.com/) installed and running, with the required models pulled:
    ```bash
    ollama pull qwen2.5-coder:14b
    ollama pull llama3.2-vision
    ollama pull nomic-embed-text   # used for agent retrieval embeddings
    ```

### 2. Installation
```bash
git clone https://github.com/MOulikSharma7/RevOps_2.0.git
cd RevOps_2.0
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration
Copy the example environment file and adjust as needed:
```bash
cp .env.example .env             # On Windows: copy .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URL` | `mongodb://localhost:27017/` | MongoDB connection string |
| `MONGODB_NAME` | `RevOps_` | Database name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_SUPERVISOR_MODEL` | `qwen2.5-coder:14b` | Supervisor / DB query LLM |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for agent retrieval |

`.env` is loaded automatically (via `python-dotenv`) when you run `main.py`.

### 4. Run
```bash
python main.py
```
This starts an interactive prompt. Type a request (e.g. *"Analyze the recent failures on genaidevassetv1"*) and the orchestrator will route through the relevant agents. Type `exit` or `quit` to leave.

## 🧪 Testing

The test suite is fully offline — it does not require MongoDB, Ollama, or the heavy LangGraph/LlamaIndex stack. It covers the supervisor routing logic, prompt/config consistency, LLM-response JSON parsing, the automation scaffolder, and the orchestrator state contract.

```bash
pip install -r requirements-dev.txt
pytest
```

Tests that need optional heavy dependencies (e.g. building a real LangGraph) are skipped automatically when those packages are absent.

## ⚠️ Status & Known Limitations

* **Automation Agent** produces a deterministic, commented script *scaffold* from the NBA plan — it does not yet emit fully executable remediation scripts.
* **`execute_agent`** and **`RAG_agent`** are placeholders that return clearly-labelled simulated responses.
* **`evaluate_nba`** (DeepEval validation) is not wired up and raises `NotImplementedError` if invoked; it is excluded from the NBA graph.
* Running the full pipeline requires a reachable MongoDB instance and a running Ollama server with the models above.

## 📄 License

Released under the MIT License.
