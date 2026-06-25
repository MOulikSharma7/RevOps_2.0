"""RevOps 2.0 entry point.

Adds ``src/`` to the import path, loads ``.env`` (if python-dotenv is present),
builds the orchestrator graph, and runs an interactive REPL against it.

Requires a running Ollama server and MongoDB instance (see README).
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from orchestrator_agent.graph import build_intent_graph


def main() -> None:
    app = build_intent_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("RevOps 2.0 Orchestrator. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"\nRevOps: {result['messages'][-1]['content']}\n")


if __name__ == "__main__":
    main()
