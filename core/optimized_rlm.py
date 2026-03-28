"""
core/optimized_rlm.py - Hybrid deterministic RLM
=================================================

This module combines:
  - deterministic recursive chunking and budgets
  - direct text summarisation/synthesis calls
  - optional trajectory logging

It is designed as a more reliable alternative to the REPL-driven RLM flow.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from core.document import Document
from core.prompts import build_leaf_prompt, build_synthesis_prompt
from core.rlm_system import RLMResult


@dataclass
class OptimizedRLMConfig:
    """Configuration for the optimized deterministic RLM mode."""

    max_depth: int = 6
    leaf_chunk_words: int = 800
    min_leaf_chunk_words: int = 200
    chunk_overlap_words: int = 150
    max_llm_calls: int = 100
    max_elapsed_sec: float = 600.0
    enable_final_synthesis: bool = True
    log_path: Optional[str] = None
    focused_excerpt_words: int = 120
    max_focused_excerpts: int = 6
    max_relevant_chunks: int = 6
    min_chunk_score: int = 1


class TrajectoryLogger:
    """Append-only JSONL logger for tracing recursive execution."""

    def __init__(self, path: Optional[str]):
        self.path = path
        if self.path:
            folder = os.path.dirname(os.path.abspath(self.path))
            if folder:
                os.makedirs(folder, exist_ok=True)

    def log(self, event: str, **payload) -> None:
        if not self.path:
            return
        record = {"event": event, **payload}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")


class OptimizedRLMSystem:
    """
    Deterministic recursive RLM that uses the LLM only for leaf summarisation
    and answer synthesis.
    """

    def __init__(
        self,
        llm,
        recursive_llm=None,
        config: Optional[OptimizedRLMConfig] = None,
        verbose: bool = False,
    ):
        self.llm = llm
        self.recursive_llm = recursive_llm or llm
        self.config = config or OptimizedRLMConfig()
        self.verbose = verbose
        self.logger = TrajectoryLogger(self.config.log_path)

        self._start_time: float = 0.0
        self._llm_calls_at_start: int = 0
        self._max_depth_reached: int = 0
        self._had_failure: bool = False
        self._failure_reason: Optional[str] = None

    def run(self, document: Document, question: str) -> RLMResult:
        self._start_time = time.time()
        self._llm_calls_at_start = self._combined_call_count()
        self._max_depth_reached = 0
        self._had_failure = False
        self._failure_reason = None

        self.logger.log(
            "run_start",
            document=document.name,
            words=document.word_count,
            question=question,
        )

        answer = self._process(document, question, depth=0)
        elapsed = time.time() - self._start_time
        llm_calls = self._combined_call_count() - self._llm_calls_at_start

        self.logger.log(
            "run_end",
            succeeded=not self._had_failure,
            llm_calls=llm_calls,
            elapsed_sec=elapsed,
        )

        return RLMResult(
            answer=answer,
            llm_calls=llm_calls,
            elapsed_sec=elapsed,
            max_depth=self._max_depth_reached,
            succeeded=not self._had_failure,
            failure=self._failure_reason,
        )

    def _process(self, document: Document, question: str, depth: int) -> str:
        if depth > self._max_depth_reached:
            self._max_depth_reached = depth

        budget_error = self._budget_failure()
        if budget_error:
            self._mark_failure(budget_error)
            return "[budget failure: {}]".format(budget_error)

        if depth > self.config.max_depth:
            msg = "recursion exceeded max_depth={}".format(self.config.max_depth)
            self._mark_failure(msg)
            return "[depth limit: {}]".format(msg)

        if self.verbose:
            print("  [OPT] depth={} doc={!r} words={:,}".format(depth, document.name[:50], len(document)))

        self.logger.log(
            "node_start",
            depth=depth,
            document=document.name,
            words=document.word_count,
        )

        target = min(self.config.leaf_chunk_words, self.llm.context_window)
        if document.word_count <= target:
            answer = self._summarize_leaf(document, question, depth, target_words=target)
        else:
            chunks = self._split_with_overlap(document, target, self.config.chunk_overlap_words)
            chunks = self._prioritize_chunks(chunks, question)
            child_answers = [self._process(chunk, question, depth + 1) for chunk in chunks]
            answer = self._synthesize_answers(
                question=question,
                partial_answers=child_answers,
                document_name=document.name,
                depth=depth,
            )

        self.logger.log(
            "node_end",
            depth=depth,
            document=document.name,
            answer_preview=answer[:200],
        )
        return answer

    def _summarize_leaf(
        self,
        document: Document,
        question: str,
        depth: int,
        target_words: int,
    ) -> str:
        content = self._select_leaf_content(document, question)
        prompt = build_leaf_prompt(
            document_name=document.name,
            word_count=document.word_count,
            question=question,
            content=content,
        )
        self.logger.log("leaf_llm_call", depth=depth, document=document.name)
        try:
            return self._llm_for_depth(depth).complete_text(prompt).strip()
        except RuntimeError as exc:
            if self._is_prompt_limit_error(exc) and document.word_count > self.config.min_leaf_chunk_words:
                smaller_target = max(
                    self.config.min_leaf_chunk_words,
                    min(target_words // 2, document.word_count - 1),
                )
                if smaller_target < document.word_count:
                    self.logger.log(
                        "leaf_backoff_split",
                        depth=depth,
                        document=document.name,
                        old_words=document.word_count,
                        new_target=smaller_target,
                    )
                    parts = self._split_with_overlap(
                        document=document,
                        target_words=smaller_target,
                        overlap_words=min(self.config.chunk_overlap_words, max(0, smaller_target // 8)),
                    )
                    partial_answers = [
                        self._summarize_leaf(part, question, depth + 1, smaller_target)
                        for part in parts
                    ]
                    return self._synthesize_answers(
                        question=question,
                        partial_answers=partial_answers,
                        document_name=document.name,
                        depth=depth,
                    )
            raise

    def _synthesize_answers(
        self,
        question: str,
        partial_answers: list[str],
        document_name: str,
        depth: int,
    ) -> str:
        cleaned = [answer.strip() for answer in partial_answers if answer and answer.strip()]
        if not cleaned:
            return "No relevant information found."
        if len(cleaned) == 1:
            return cleaned[0]

        if not self.config.enable_final_synthesis:
            return "\n\n".join(cleaned)

        prompt = build_synthesis_prompt(
            document_name=document_name,
            question=question,
            partial_answers=cleaned,
        )
        self.logger.log("synthesis_llm_call", depth=depth, document=document_name, count=len(cleaned))
        return self._llm_for_depth(depth).complete_text(prompt).strip()

    def _split_with_overlap(self, document: Document, target_words: int, overlap_words: int) -> list[Document]:
        words = document.word_count
        if words <= target_words:
            return [document]

        stride = max(1, target_words - max(0, overlap_words))
        num_chunks = max(2, math.ceil(max(1, words - overlap_words) / stride))
        chunks = []
        start = 0
        for _ in range(num_chunks):
            end = min(start + target_words, words)
            chunks.append(document.slice(start, end))
            if end >= words:
                break
            start += stride
        return chunks

    def _select_leaf_content(self, document: Document, question: str) -> str:
        """
        Prefer focused excerpts for lookup-style questions so we do not waste
        prompt budget on the entire chunk when only a small local region matters.
        """
        if self._is_lookup_question(question):
            excerpts = self._extract_focused_excerpts(document, question)
            if excerpts:
                return "\n\n".join(excerpts)
        return document.peek(0, document.word_count)

    def _prioritize_chunks(self, chunks: list[Document], question: str) -> list[Document]:
        """
        Rank chunks by lexical relevance for lookup-style or fact-centric queries.
        Falls back to the full ordered list when relevance is weak.
        """
        if len(chunks) <= self.config.max_relevant_chunks:
            return chunks

        terms = self._query_terms(question)
        if not terms:
            return chunks

        scored = []
        for index, chunk in enumerate(chunks):
            score = self._chunk_relevance_score(chunk, terms)
            scored.append((score, index, chunk))

        strong = [item for item in scored if item[0] >= self.config.min_chunk_score]
        if not strong:
            return chunks

        strong.sort(key=lambda item: (-item[0], item[1]))
        selected = [item[2] for item in strong[: self.config.max_relevant_chunks]]
        selected.sort(key=lambda chunk: chunks.index(chunk))
        self.logger.log(
            "chunk_prioritization",
            selected=len(selected),
            total=len(chunks),
            question=question,
        )
        return selected

    def _extract_focused_excerpts(self, document: Document, question: str) -> list[str]:
        terms = self._query_terms(question)
        if not terms:
            return []
        words = document.content.split()
        lowered = [word.lower() for word in words]
        excerpts = []
        seen_ranges = set()

        for idx, token in enumerate(lowered):
            if not any(term in token for term in terms):
                continue
            half_window = max(20, self.config.focused_excerpt_words // 2)
            start = max(0, idx - half_window)
            end = min(len(words), idx + half_window)
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
            if len(excerpts) >= self.config.max_focused_excerpts:
                break
        return excerpts

    @staticmethod
    def _chunk_relevance_score(document: Document, terms: list[str]) -> int:
        lowered = document.content.lower()
        score = 0
        for term in terms:
            if term in lowered:
                score += lowered.count(term) * max(1, min(4, len(term) // 4))
        return score

    @staticmethod
    def _is_lookup_question(question: str) -> bool:
        lower = question.lower()
        markers = ("email", "e-mail", "phone", "contact", "address", "who is", "what is the")
        return any(marker in lower for marker in markers)

    @staticmethod
    def _query_terms(question: str) -> list[str]:
        stopwords = {
            "what", "which", "when", "where", "who", "whom", "whose", "is", "the",
            "of", "for", "and", "to", "in", "a", "an", "dr", "smt", "mr", "mrs",
            "ms", "email", "e", "mail",
        }
        terms = []
        for raw in re.findall(r"[A-Za-z0-9@._-]+", question.lower()):
            if len(raw) < 3 or raw in stopwords:
                continue
            terms.append(raw)
        return terms

    @staticmethod
    def _is_prompt_limit_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "prompt tokens limit exceeded" in msg or "maximum context length" in msg

    def _budget_failure(self) -> Optional[str]:
        elapsed = time.time() - self._start_time
        llm_calls = self._combined_call_count() - self._llm_calls_at_start
        if elapsed > self.config.max_elapsed_sec:
            return "elapsed time exceeded {:.1f}s".format(self.config.max_elapsed_sec)
        if llm_calls >= self.config.max_llm_calls:
            return "llm calls exceeded {}".format(self.config.max_llm_calls)
        return None

    def _llm_for_depth(self, depth: int):
        return self.llm if depth == 0 else self.recursive_llm

    def _combined_call_count(self) -> int:
        total = self.llm.call_count
        if self.recursive_llm is not self.llm:
            total += self.recursive_llm.call_count
        return total

    def _mark_failure(self, reason: str) -> None:
        self._had_failure = True
        if self._failure_reason is None:
            self._failure_reason = reason
