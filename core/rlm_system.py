"""
core/rlm_system.py - Layer 4: The Orchestrator
================================================

RLMSystem is the top-level coordinator for the Recursive Language Model.

Public API:
    rlm = RLMSystem(llm, max_depth=6, max_turns_per_node=5, verbose=True)
    result = rlm.run(document, question)   -> RLMResult

Algorithm:
    run(document, question):
        if document.fits_in_window(context_window):
            -> call _call(document, question, depth=0)  # goes straight to REPL
        else:
            -> call _call(document, question, depth=0)  # REPL will split

    _call(document, question, depth):
        if depth > max_depth: return "[depth limit reached]"
        build REPLNamespace with sub_call_handler = lambda d,q: _call(d, q, depth+1)
        run REPLExecutor.run(document, question, sub_call_handler)
        return result.answer

The recursion is established through the sub_call_handler closure:
  - When the LLM-generated code calls sub_call(chunk, Q)
  - REPLNamespace routes it to sub_call_handler(chunk, Q)
  - Which calls _call(chunk, Q, depth+1)
  - _call creates a new REPLExecutor for the chunk and the cycle repeats

RLMResult fields:
    answer          -- The final synthesised answer string
    llm_calls       -- Total LLM API calls made across the whole tree
    elapsed_sec     -- Wall-clock seconds from run() start to finish
    max_depth_reached -- Deepest recursion level used
    succeeded       -- True unless depth-limit or timeout killed the run
    failure_reason  -- Human-readable reason on failure, else None
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from core.document import Document
from core.repl import REPLExecutor


# ---------------------------------------------------------------------------
# Result datatype
# ---------------------------------------------------------------------------

@dataclass
class RLMResult:
    """
    The complete result of one RLMSystem.run() call.

    Attributes:
        answer:            Final synthesised answer.
        llm_calls:         Total LLM API calls.
        elapsed_sec:       Wall-clock seconds.
        max_depth_reached: Deepest recursion level used.
        succeeded:         True if final() was reached without hitting limits.
        failure_reason:    Populated on failure, None on success.
    """
    answer: str
    llm_calls: int
    elapsed_sec: float
    max_depth_reached: int
    succeeded: bool
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# RLMSystem
# ---------------------------------------------------------------------------

class RLMSystem:
    """
    Orchestrates the full RLM recursive document processing pipeline.

    Args:
        llm:               Any BaseLanguageModel instance (real or mock).
        max_depth:         Maximum recursion depth before hard stop (default 6).
        max_turns_per_node:Max REPL turns per document node (default 5).
        verbose:           Print progress to stdout.
    """

    def __init__(
        self,
        llm,
        max_depth: int = 6,
        max_turns_per_node: int = 5,
        verbose: bool = False,
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.max_turns_per_node = max_turns_per_node
        self.verbose = verbose

        # State reset on each run()
        self._llm_calls_at_start: int = 0
        self._max_depth_reached: int = 0
        self._start_time: float = 0.0
        self._had_failure: bool = False
        self._failure_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, document: Document, question: str) -> RLMResult:
        """
        Process *document* with *question* using the full RLM pipeline.

        Args:
            document: A Document loaded via load_document_from_folder().
            question: The user's natural-language question.

        Returns:
            RLMResult with the final answer and run statistics.
        """
        # Reset per-run state
        self._llm_calls_at_start = self.llm.call_count
        self._max_depth_reached = 0
        self._start_time = time.time()
        self._had_failure = False
        self._failure_reason = None

        if self.verbose:
            wc = len(document)
            cw = self.llm.context_window
            fits = document.fits_in_window(cw)
            print("  [RLM] Document: {!r} ({:,} words)".format(document.name, wc))
            print("  [RLM] Context window: {:,} words | Fits: {}".format(cw, fits))
            if not fits:
                k = max(2, (wc + cw - 1) // cw)
                print("  [RLM] Expected initial split into ~{} chunks".format(k))
            print()

        # The main recursive call (depth=0)
        answer = self._call(document, question, depth=0)

        elapsed = time.time() - self._start_time
        llm_calls = self.llm.call_count - self._llm_calls_at_start

        succeeded = not self._had_failure
        failure_reason = self._failure_reason

        return RLMResult(
            answer=answer,
            llm_calls=llm_calls,
            elapsed_sec=elapsed,
            max_depth_reached=self._max_depth_reached,
            succeeded=succeeded,
            failure_reason=failure_reason,
        )

    # ------------------------------------------------------------------
    # Recursive worker
    # ------------------------------------------------------------------

    def _call(self, document: Document, question: str, depth: int) -> str:
        """
        Process one document node at a given recursion depth.

        This is the recursive core of RLM.  It creates a new REPLExecutor
        for every node so that each has a clean, independent history.

        The sub_call_handler closure is what makes the recursion work:
          When the LLM-written code calls sub_call(chunk, Q), the namespace
          routes it here as _call(chunk, Q, depth+1).

        Args:
            document: The Document chunk to process.
            question: The question to answer.
            depth:    Current recursion depth (0 = root).

        Returns:
            Answer string (may be a failure marker on depth/timeout).
        """
        # Track deepest depth
        if depth > self._max_depth_reached:
            self._max_depth_reached = depth

        # Hard stop — prevents infinite recursion
        if depth > self.max_depth:
            msg = "[depth limit: recursion exceeded max_depth={}]".format(self.max_depth)
            self._mark_failure(msg)
            if self.verbose:
                print("  [RLM] DEPTH LIMIT at depth {} for {!r}".format(depth, document.name))
            return msg

        if self.verbose:
            print("  [RLM] _call(depth={}, doc={!r}, words={:,})".format(
                depth, document.name[:50], len(document)
            ))

        # Create the sub_call handler that wires recursion
        def sub_call_handler(child_doc: Document, child_q: str) -> str:
            return self._call(child_doc, child_q, depth + 1)

        # Create and run the REPL executor for this node
        executor = REPLExecutor(
            llm=self.llm,
            max_turns=self.max_turns_per_node,
            verbose=self.verbose,
        )
        result = executor.run(
            document=document,
            question=question,
            sub_call_handler=sub_call_handler,
        )

        if self.verbose:
            status = "OK" if result.succeeded else "FAILED"
            print("  [RLM] _call(depth={}) -> {} | {} LLM calls | answer: {!r}".format(
                depth,
                status,
                result.llm_calls,
                result.answer[:60] + ("..." if len(result.answer) > 60 else ""),
            ))

        if not result.succeeded:
            self._mark_failure(result.failure_reason or result.answer)

        return result.answer

    def _mark_failure(self, reason: str) -> None:
        """Record the first failure seen anywhere in the recursion tree."""
        self._had_failure = True
        if self._failure_reason is None:
            self._failure_reason = reason
