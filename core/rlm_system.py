import time
import re
from dataclasses import dataclass
from typing import Optional

from .document import Document
from .llm import BaseLLM
from .repl import REPLExecutor, REPLNamespace


@dataclass
class RLMResult:
    answer: str
    llm_calls: int
    elapsed_sec: float
    max_depth: int
    succeeded: bool
    failure: Optional[str] = None
    repl_trace: Optional[list[str]] = None

    def display(self):
        sep = "═" * 60
        status = "✓ SUCCESS" if self.succeeded else "✗ FAILED"
        print(f"\n{sep}")
        print(f"  {status}")
        print(f"{sep}")
        print(f"  Answer:\n")
        words = self.answer.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 80:
                print(f"  {line}")
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            print(f"  {line}")
        print(f"\n{sep}")
        print(f"  LLM calls made  : {self.llm_calls}")
        print(f"  Time elapsed    : {self.elapsed_sec:.1f}s")
        print(f"  Max depth       : {self.max_depth}")
        if self.failure:
            print(f"  Failure reason  : {self.failure}")
        print(sep)


class RLMSystem:
    """
    The complete RLM system (Zhang et al. 2026).

    THIS IS PURE RLM — not λ-RLM.

    What it does:
      - Stores the document externally (never in LLM context)
      - Lets the LLM write Python code to decompose the problem
      - Recursively calls itself on sub-documents via sub_call()
      - Returns the final answer when the LLM calls final()

    What it does NOT do (these are λ-RLM's contributions):
      - Does NOT pre-compute k*, depth, cost before execution
      - Does NOT guarantee termination
      - Does NOT use a typed combinator library
      - Does NOT separate planning from execution

    Known failure modes (see repl.py for details):
      1. LLM writes broken Python → exec() error
      2. LLM calls sub_call(P, q) with same-size doc → infinite recursion
      3. LLM never calls final() → runs until max_turns
      4. LLM decides k unpredictably → cost unknown in advance
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_depth: int = 6,
        max_turns_per_node: int = 5,
        verbose: bool = True,
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.max_turns = max_turns_per_node
        self.verbose = verbose
        self._max_depth_seen = 0
        self._repl_trace: list[str] = []

    def run(self, document: Document, question: str) -> RLMResult:
        """Entry point. Call this with your loaded Document."""
        print(f"\n{'#'*60}")
        print(f"# RLM System (Standard — open-ended REPL loop)")
        print(f"# Document : '{document.name}' ({len(document):,} words)")
        print(f"# Window   : {self.llm.config.context_window:,} words")
        print(f"# Fits?    : {document.fits_in_window(self.llm.config.context_window)}")
        print(f"# Question : {question}")
        print(f"{'#'*60}")

        self._max_depth_seen = 0
        self._repl_trace = []
        calls_before = self.llm.call_count
        t0 = time.time()

        answer = self._call(document, question, depth=0)
        elapsed = time.time() - t0

        result = RLMResult(
            answer=answer,
            llm_calls=self.llm.call_count - calls_before,
            elapsed_sec=elapsed,
            max_depth=self._max_depth_seen,
            succeeded=bool(answer and "error" not in answer.lower()[:20]),
            repl_trace=list(self._repl_trace),
        )
        result.display()
        print(self.llm.stats())
        return result

    def _call(self, doc: Document, question: str, depth: int) -> str:
        """
        One recursive node in the call tree.

        Called at depth=0 from run(), and at depth=1,2,... from
        exec()'d sub_call() code via the injected lambda below.
        """
        self._max_depth_seen = max(self._max_depth_seen, depth)

        if depth > self.max_depth:
            return f"(max recursion depth {self.max_depth} reached)"

        if depth == 0:
            doc = self._maybe_focus_root_document(doc, question)

        ns = REPLNamespace(
            document=doc,
            question=question,
            sub_call_handler=lambda d, q: self._call(d, q, depth + 1),
        )

        executor = REPLExecutor(llm=self.llm, max_turns=self.max_turns)
        result = executor.run(doc, question, ns, depth=depth)
        self._repl_trace.append(
            "### Node depth={} doc='{}'\n{}".format(
                depth,
                doc.name,
                "\n\n".join(result.history) if result.history else "# No generated code captured",
            )
        )
        return result.answer

    def _maybe_focus_root_document(self, doc: Document, question: str) -> Document:
        """
        For lookup-style questions, shrink the root REPL document to a small set
        of relevant excerpts so local models do not spend minutes planning over
        a huge preview-only node.
        """
        if doc.fits_in_window(self.llm.config.context_window):
            return doc
        if not self._is_lookup_question(question):
            return doc

        excerpts = self._extract_focused_excerpts(doc, question)
        if not excerpts:
            return doc

        excerpt_text = "\n\n".join(excerpts)
        focused_doc = Document(
            content=excerpt_text,
            name=f"{doc.name}[focused excerpts]",
            parent_name=doc.name,
            depth=doc.depth + 1,
        )
        if self.verbose:
            print(
                "[RLM] Using focused excerpt document for lookup-style question "
                f"({focused_doc.word_count:,} words)"
            )
        return focused_doc

    @staticmethod
    def _is_lookup_question(question: str) -> bool:
        lower = question.lower()
        markers = ("email", "e-mail", "phone", "contact", "address", "who is", "what is the")
        return any(marker in lower for marker in markers)

    @staticmethod
    def _query_terms(question: str) -> list[str]:
        stopwords = {
            "what", "which", "when", "where", "who", "whom", "whose", "is", "the",
            "of", "for", "and", "to", "in", "a", "an", "doc", "document", "mentioned",
        }
        terms = []
        for raw in re.findall(r"[A-Za-z0-9@._-]+", question.lower()):
            if len(raw) < 3 or raw in stopwords:
                continue
            terms.append(raw)
        return terms

    def _extract_focused_excerpts(self, doc: Document, question: str) -> list[str]:
        terms = self._query_terms(question)
        if not terms:
            return []
        words = doc.content.split()
        lowered = [word.lower() for word in words]
        excerpts = []
        seen_ranges = set()

        for idx, token in enumerate(lowered):
            if not any(term in token for term in terms):
                continue
            start = max(0, idx - 50)
            end = min(len(words), idx + 70)
            key = (start, end)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            excerpt = " ".join(words[start:end]).strip()
            if excerpt:
                excerpts.append(
                    "EXCERPT {} (words {}-{}):\n{}".format(
                        len(excerpts) + 1,
                        start,
                        end,
                        excerpt,
                    )
                )
            if len(excerpts) >= 6:
                break
        return excerpts
