# 🛠️ RevOps 2.0: Autonomous Multi-Agent IT Remediation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange)](https://python.langchain.com/docs/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-black)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**RevOps** is an autonomous, multi-agent AI framework designed to automate IT operations, root cause analysis (RCA), and incident remediation. Built on **LangGraph**, it coordinates specialized sub-agents to independently query databases, fetch server logs via SSH, diagnose errors, and generate actionable technical solutions.

---

## 🚀 Architecture & Sub-Agents

The system is managed by an **Orchestrator Agent** that uses semantic routing (via LlamaIndex) and a Supervisor LLM to dynamically select and execute the appropriate sub-agents based on the active diagnostic state.

* **🧠 Orchestrator Agent**: The central supervisor. Maintains state memory, analyzes user intent, and enforces strict execution guardrails to prevent infinite loops.
* **🗄️ DB Agent**: Translates natural language into MongoDB queries (`qwen2.5-coder`), executes them, and returns infrastructure context (e.g., Server IPs, credentials).
* **📡 Log Retriever Agent**: Establishes SSH connections using `paramiko` to retrieve live system logs (`journalctl`) for specific nodes.
* **🔍 RCA Agent (Root Cause Analysis)**: Analyzes system exceptions against retrieved log files using multimodal LLMs (`llama3.2-vision`) to pinpoint exact failure points.
* **💡 NBA Agent (Next Best Action)**: Consumes the RCA report to generate a chronological, actionable remediation plan.
* **⚙️ Automation Agent** *(In Development)*: Translates NBA plans into executable Python/Bash scripts for auto-remediation.

## 💻 Tech Stack

* **Framework**: [LangGraph](https://python.langchain.com/docs/langgraph) & LangChain
* **Local LLMs**: [Ollama](https://ollama.com/) (`qwen2.5-coder:14b`, `llama3.2-vision`)
* **Database**: MongoDB (via `pymongo`)
* **Routing & Retrieval**: LlamaIndex VectorStore
* **Infrastructure**: `paramiko` (SSH automation)

## 🛠️ Quickstart

### 1. Prerequisites
* Python 3.10+
* MongoDB running locally or remotely
* Ollama installed and running with required models:
    ```bash
    ollama run qwen2.5-coder:14b
    ollama run llama3.2-vision
    ```

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/FieldOps_2.0.git](https://github.com/YOUR_USERNAME/FieldOps_2.0.git)
cd FieldOps_2.0
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
