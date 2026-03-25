"""
Reusable package-style API for document and raw-context completion.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from core.document import Document
from core.optimized_rlm import OptimizedRLMConfig, OptimizedRLMSystem


@dataclass
class RLMConfig:
    """High-level API config for package-style access."""

    model: Optional[str] = None
    recursive_model: Optional[str] = None
    mode: str = "optimized"
    max_depth: int = 6
    leaf_chunk_words: int = 800
    min_leaf_chunk_words: int = 200
    chunk_overlap_words: int = 150
    max_llm_calls: int = 100
    max_elapsed_sec: float = 600.0
    enable_final_synthesis: bool = True
    log_path: Optional[str] = None


class RLM:
    """
    Upstream-style reusable API layered on top of the local document pipeline.
    """

    def __init__(self, llm, recursive_llm=None, config: Optional[RLMConfig] = None, verbose: bool = False):
        self.llm = llm
        self.recursive_llm = recursive_llm or llm
        self.config = config or RLMConfig()
        self.verbose = verbose

    def complete(self, context: str, question: str, document_name: str = "context.txt"):
        """Run optimized RLM over a raw string context."""
        document = Document(name=document_name, content=context)
        system = OptimizedRLMSystem(
            llm=self.llm,
            recursive_llm=self.recursive_llm,
            config=OptimizedRLMConfig(
                max_depth=self.config.max_depth,
                leaf_chunk_words=self.config.leaf_chunk_words,
                min_leaf_chunk_words=self.config.min_leaf_chunk_words,
                chunk_overlap_words=self.config.chunk_overlap_words,
                max_llm_calls=self.config.max_llm_calls,
                max_elapsed_sec=self.config.max_elapsed_sec,
                enable_final_synthesis=self.config.enable_final_synthesis,
                log_path=self.config.log_path,
            ),
            verbose=self.verbose,
        )
        return system.run(document, question)

    async def acomplete(self, context: str, question: str, document_name: str = "context.txt"):
        """Async wrapper so callers can compose multiple runs concurrently."""
        return await asyncio.to_thread(self.complete, context, question, document_name)
