"""Centralised LlamaIndex/Ollama runtime configuration.

The orchestrator uses ``Settings.llm`` (the supervisor LLM) and
``Settings.embed_model`` (for the semantic agent retriever). This singleton
wires both to a local Ollama server exactly once.
"""

import os

from llama_index.core import Settings


class clsLlamaIndexManager:
    """Singleton that configures LlamaIndex global ``Settings`` for Ollama."""

    _instance = None

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm_model = os.getenv("OLLAMA_SUPERVISOR_MODEL", "qwen2.5-coder:14b")
        self.embed_model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self._configure()

    @classmethod
    def get_instance(cls) -> "clsLlamaIndexManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _configure(self) -> None:
        # Imported lazily so this module imports even if the optional Ollama
        # integration packages are absent; configuration fails loudly only when
        # an instance is actually requested.
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama

        Settings.llm = Ollama(
            model=self.llm_model,
            base_url=self.base_url,
            request_timeout=120.0,
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=self.embed_model_name,
            base_url=self.base_url,
        )
        self.llm = Settings.llm
        self.embed_model = Settings.embed_model
