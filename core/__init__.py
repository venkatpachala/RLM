"""
core/__init__.py - Public exports for the RLM core package
"""
from core.document import Document, load_document_from_folder
from core.llm import OpenAICompatibleLLM, OllamaLLM, OpenRouterLLM
from core.api import RLM, RLMConfig
from core.repl import REPLExecutor, REPLNamespace
from core.rlm_system import RLMSystem, RLMResult

__all__ = [
    "Document",
    "load_document_from_folder",
    "OpenAICompatibleLLM",
    "OllamaLLM",
    "OpenRouterLLM",
    "RLM",
    "RLMConfig",
    "REPLExecutor",
    "REPLNamespace",
    "RLMSystem",
    "RLMResult",
]
