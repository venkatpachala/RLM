

from dataclasses import dataclass, field
from typing import Callable, Optional

from .document import Document



class REPLNamespace:
    """
    The 'global scope' that exec(generated_code) runs inside.

    When the LLM writes:
        chunks = split(P, 4)

    Python resolves 'split' and 'P' from this namespace dict.
    We control exactly what's available — nothing else.

    Why this matters:
      - Safety: LLM can't call os.system(), import random, etc.
      - Observability: every sub_call() and final() is captured here
      - Injectability: sub_call_handler is passed in from RLMSystem,
        creating the recursive wiring
    """

    def __init__(
        self,
        document: Document,
        question: str,
        sub_call_handler: Callable[[Document, str], str],
        max_k: int = 8,
    ):
        self.document = document
        self.question = question
        self._handler = sub_call_handler
        self._max_k = max_k

        # Written by the LLM calling these functions
        self._answer: Optional[str] = None
        self._done: bool = False


    def split(self, doc: Document, k: int) -> list[Document]:
        """
        Split doc into k equal chunks.
        The LLM decides k in RLM. λ-RLM always uses k*=2.
        """
        if k < 1:
            raise ValueError(f"split(P, {k}): k must be ≥ 1")
        k = min(k, self._max_k)
        chunks = doc.split(k)
        print(f"      [SPLIT] {doc.name} → {k} chunks (~{len(doc)//k} words each)")
        return chunks

    def peek(self, doc: Document, start: int, end: int) -> str:
        """Cheap inspection. Reads a slice without an LLM call."""
        result = doc.peek(start, end)
        print(f"      [PEEK] {doc.name}[{start}:{end}] = '{result[:50]}...'")
        return result

    def sub_call(self, doc: Document, question: str) -> str:
        """
        Recursive invocation. Calls RLMSystem._call(doc, question, depth+1).
        This is the heart of the recursion — injected via sub_call_handler.

        ⚠ RLM failure mode: if doc is same size as parent → infinite recursion.
        λ-RLM eliminates this: SPLIT always decreases size by factor k*=2.
        """
        print(f"      [SUB_CALL] → {doc.name} ({len(doc)} words)")
        return self._handler(doc, question)

    def merge(self, answers: list) -> str:
        """
        Combine partial answers. Filters empty/negative results.
        In a smarter system this would be another LLM call.
        """
        if not answers:
            return "No content found."
        kept = []
        for a in answers:
            a_str = str(a).strip()
            if a_str and len(a_str) > 5 and not a_str.lower().startswith("no relevant"):
                kept.append(a_str)
        if not kept:
            return "No relevant findings in any section."
        return "\n\n".join(kept)

    def final(self, answer: str):
        """
        Signals the REPL loop to stop and return this answer.
        ⚠ RLM failure mode: if LLM never calls this, loop runs until timeout.
        λ-RLM eliminates this: Φ always terminates (Theorem 1).
        """
        self._answer = str(answer)
        self._done = True
        print(f"      [FINAL] '{str(answer)[:100]}...'")

    def as_dict(self) -> dict:
        """Build the namespace dict that exec() uses as its globals."""
        return {
            "P":        self.document,
            "question": self.question,
            "split":    self.split,
            "peek":     self.peek,
            "sub_call": self.sub_call,
            "merge":    self.merge,
            "final":    self.final,
            "__builtins__": {
                "len": len, "range": range, "list": list, "str": str,
                "int": int, "float": float, "print": print,
                "enumerate": enumerate, "zip": zip,
                "max": max, "min": min, "True": True, "False": False, "None": None,
            },
        }


@dataclass
class REPLResult:
    answer: str
    turns: int
    succeeded: bool
    failure: Optional[str] = None
    history: list[str] = field(default_factory=list)


class REPLExecutor:
    """
    Runs one node of the RLM recursion tree.

    For each turn:
      1. Build prompt (doc preview + metadata + question + history)
      2. Call LLM → get code
      3. exec(code) in namespace
      4. If final() was called → return answer
      5. Else add turn to history, repeat

    ⚠ No built-in termination guarantee — only the LLM calling final() stops it.
    We add max_turns as a safety net.
    """

    def __init__(self, llm, max_turns: int = 6):
        self.llm = llm
        self.max_turns = max_turns

    def run(self, doc: Document, question: str,
            ns: REPLNamespace, depth: int = 0) -> REPLResult:

        indent = "  " * depth
        history: list[str] = []

        print(f"\n{indent}{'─'*55}")
        print(f"{indent}RLM Node  depth={depth}  doc='{doc.name}'  words={len(doc)}")
        print(f"{indent}{'─'*55}")

        for turn in range(self.max_turns):
            prompt = self._prompt(doc, question, history)

            print(f"{indent}[Turn {turn+1}/{self.max_turns}] Calling LLM...")
            code = self.llm.generate(prompt)

            output, error = self._exec(code, ns.as_dict())

            entry = f"[Turn {turn+1}]\n{code}\n"
            entry += f"# OK" if not error else f"# ERROR: {error}"
            history.append(entry)

            if error:
                print(f"{indent}  ⚠ Code error: {error}")
            if ns._done:
                print(f"{indent}✓ Done in {turn+1} turn(s)")
                return REPLResult(
                    answer=ns._answer,
                    turns=turn+1,
                    succeeded=True,
                    history=history,
                )

        fallback = ns._answer or "No answer produced (max turns reached)."
        print(f"{indent}✗ Timeout after {self.max_turns} turns")
        return REPLResult(
            answer=fallback, turns=self.max_turns, succeeded=False,
            failure=f"LLM never called final() in {self.max_turns} turns",
            history=history,
        )

    def _prompt(self, doc: Document, question: str, history: list[str]) -> str:
        """
        What enters the LLM's context window each turn.

        Critically: the full document is NEVER here.
        Only: 150-word preview + word count + question + last 3 turns of history.
        """
        preview = doc.peek(0, 150)
        history_str = "\n\n".join(history[-3:]) if history else "(first turn)"

        return f"""DOCUMENT: '{doc.name}'
DOCUMENT WORDS: {len(doc)}
CONTEXT WINDOW: {self.llm.config.context_window} words
FITS IN WINDOW: {'YES — answer directly' if doc.fits_in_window(self.llm.config.context_window) else 'NO — must split'}

DOCUMENT CONTENT {'(fits — read and answer)' if doc.fits_in_window(self.llm.config.context_window) else '(preview only)'}:
{doc.content if doc.fits_in_window(self.llm.config.context_window) else preview + chr(10) + '[...document continues beyond context window...]'}

QUESTION: {question}

HISTORY:
{history_str}

Write Python code. You MUST call final("answer") when done."""

    def _exec(self, code: str, ns: dict) -> tuple[str, Optional[str]]:
        """Run LLM-generated code in the controlled namespace."""
        code = code.strip()
        for fence in ["```python", "```py", "```"]:
            if code.startswith(fence):
                code = code[len(fence):]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        try:
            exec(code, ns)
            return "OK", None
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"
