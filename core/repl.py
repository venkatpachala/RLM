"""
core/repl.py - Layer 3: The REPL Engine
=========================================

The REPL (Read-Eval-Print Loop) is the "open-ended loop" that lets the LLM
issue Python commands to process a document chunk by chunk without ever
reading the whole document at once.

Flow per turn:
  1. Build prompt  -- doc preview + word count + fits flag + question + history
  2. Call LLM      -- get Python code string
  3. safe_exec()   -- run code in REPLNamespace (restricted scope)
  4. Append history-- LLM sees any errors next turn and can self-correct
  5. Check final() -- if namespace._is_done: return answer

Key classes:
  REPLNamespace   -- The controlled exec() scope.  Exposes only the five
                     RLM functions (split, peek, sub_call, merge, final)
                     plus safe builtins.  Everything else is blocked.

  REPLExecutor    -- Runs the REPL loop for one Document node.
                     Created fresh for every _call() in RLMSystem.

  REPLResult      -- Lightweight result datatype returned from REPLExecutor.run().
"""

from __future__ import annotations

import ast
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional

from core.document import Document


# ---------------------------------------------------------------------------
# Safe builtins available to LLM-generated code
# ---------------------------------------------------------------------------

def _blocked_import(*args, **kwargs):
    raise RuntimeError(
        "import is not allowed in RLM code blocks. "
        "Use only the pre-defined functions: split, peek, sub_call, merge, final."
    )


_SAFE_BUILTINS: dict = {
    "__builtins__": {
        # Introspection
        "len":      len,
        "range":    range,
        "enumerate":enumerate,
        "zip":      zip,
        "map":      map,
        "filter":   filter,
        # Type constructors
        "str":      str,
        "int":      int,
        "float":    float,
        "bool":     bool,
        "list":     list,
        "dict":     dict,
        "tuple":    tuple,
        "set":      set,
        # Utilities
        "print":    print,
        "repr":     repr,
        "isinstance": isinstance,
        "type":     type,
        "min":      min,
        "max":      max,
        "sum":      sum,
        "abs":      abs,
        "round":    round,
        "sorted":   sorted,
        "reversed": reversed,
        "any":      any,
        "all":      all,
        # Disallowed but must exist to suppress NameErrors on common accidents
        # (they raise a clear error message instead)
        "__import__": _blocked_import,
    }
}

_ALLOWED_CALL_NAMES = {
    "split",
    "peek",
    "sub_call",
    "merge",
    "final",
    "len",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "tuple",
    "set",
    "print",
    "repr",
    "isinstance",
    "type",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "sorted",
    "reversed",
    "any",
    "all",
}

_ALLOWED_AST_NODES = (
    ast.Module,
    ast.Assign,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.comprehension,
    ast.For,
    ast.If,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Subscript,
    ast.Slice,
    ast.keyword,
    ast.Pass,
)


class UnsafeCodeError(RuntimeError):
    """Raised when LLM-generated code uses syntax outside the safe subset."""


class _SafeCodeValidator(ast.NodeVisitor):
    """Allow only the small Python subset needed for chunk orchestration."""

    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise UnsafeCodeError(
                "Disallowed syntax: {}".format(type(node).__name__)
            )
        super().generic_visit(node)

    def visit_Attribute(self, node):
        raise UnsafeCodeError("Attribute access is not allowed.")

    def visit_Import(self, node):
        raise UnsafeCodeError("import is not allowed.")

    def visit_ImportFrom(self, node):
        raise UnsafeCodeError("import is not allowed.")

    def visit_While(self, node):
        raise UnsafeCodeError("while loops are not allowed.")

    def visit_Try(self, node):
        raise UnsafeCodeError("try/except is not allowed.")

    def visit_FunctionDef(self, node):
        raise UnsafeCodeError("Function definitions are not allowed.")

    def visit_AsyncFunctionDef(self, node):
        raise UnsafeCodeError("Function definitions are not allowed.")

    def visit_ClassDef(self, node):
        raise UnsafeCodeError("Class definitions are not allowed.")

    def visit_Lambda(self, node):
        raise UnsafeCodeError("lambda is not allowed.")

    def visit_Delete(self, node):
        raise UnsafeCodeError("delete is not allowed.")

    def visit_Global(self, node):
        raise UnsafeCodeError("global is not allowed.")

    def visit_Nonlocal(self, node):
        raise UnsafeCodeError("nonlocal is not allowed.")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise UnsafeCodeError("Only direct function calls are allowed.")
        if node.func.id not in _ALLOWED_CALL_NAMES:
            raise UnsafeCodeError(
                "Call to '{}' is not allowed.".format(node.func.id)
            )
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("__"):
            raise UnsafeCodeError("Dunder names are not allowed.")
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            self._validate_assignment_target(target)
        self.visit(node.value)

    def visit_For(self, node):
        self._validate_assignment_target(node.target)
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_comprehension(self, node):
        self._validate_assignment_target(node.target)
        self.visit(node.iter)
        for clause in node.ifs:
            self.visit(clause)

    def _validate_assignment_target(self, node):
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise UnsafeCodeError("Dunder names are not allowed.")
            return
        if isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._validate_assignment_target(elt)
            return
        raise UnsafeCodeError(
            "Invalid assignment target: {}".format(type(node).__name__)
        )


# ---------------------------------------------------------------------------
# REPLNamespace — execution scope for LLM code
# ---------------------------------------------------------------------------

class REPLNamespace:
    """
    Provides the five RLM functions to exec()'d LLM code.

    The two key variables injected as globals:
      P  -- the current Document (the LLM refers to it as 'P')
      Q  -- the question string (the LLM refers to it as 'Q')

    Functions exposed:
      split(P, k)      -> list[Document]
      peek(P, s, e)    -> str
      sub_call(doc, q) -> str   (calls back into RLMSystem._call recursively)
      merge(answers)   -> str
      final(answer)    -> None  (sets _is_done=True)
    """

    def __init__(
        self,
        document: Document,
        question: str,
        sub_call_handler: Callable[[Document, str], str],
    ):
        self._document = document
        self._question = question
        self._sub_call_handler = sub_call_handler
        self._is_done: bool = False
        self._answer: str = ""
        self._sub_call_count: int = 0

    # ------------------------------------------------------------------
    # The five RLM primitives (called from exec()'d LLM code)
    # ------------------------------------------------------------------

    def _fn_split(self, doc: Document, k: int) -> List[Document]:
        """Split document *doc* into *k* equal chunks."""
        if not isinstance(doc, Document):
            raise TypeError(
                "split() first argument must be a Document, got {!r}".format(type(doc).__name__)
            )
        if not isinstance(k, int) or k < 1:
            raise ValueError(
                "split(doc, k): k must be a positive integer, got {!r}".format(k)
            )
        return doc.split(k)

    def _fn_peek(self, doc: Document, start: int, end: int) -> str:
        """Read words[start:end] of *doc* as a string."""
        if not isinstance(doc, Document):
            raise TypeError(
                "peek() first argument must be a Document, got {!r}".format(type(doc).__name__)
            )
        return doc.peek(start, end)

    def _fn_sub_call(self, doc: Document, question: str) -> str:
        """Recursively process *doc* with *question* via RLMSystem._call()."""
        if not isinstance(doc, Document):
            raise TypeError(
                "sub_call() first argument must be a Document, got {!r}".format(type(doc).__name__)
            )
        self._sub_call_count += 1
        return self._sub_call_handler(doc, str(question))

    def _fn_merge(self, answers) -> str:
        """
        Filter out blank/None answers, join the rest with newlines.
        Always returns a string — never raises.
        """
        if not hasattr(answers, "__iter__"):
            raise TypeError(
                "merge() expects an iterable of strings, got {!r}".format(type(answers).__name__)
            )
        parts = [str(a).strip() for a in answers if a and str(a).strip()]
        return "\n\n".join(parts) if parts else "No relevant information found."

    def _fn_final(self, answer) -> None:
        """Store *answer* and signal the REPL loop to stop."""
        self._answer = str(answer).strip()
        self._is_done = True

    # ------------------------------------------------------------------
    # Build the globals dict for exec()
    # ------------------------------------------------------------------

    def as_exec_dict(self) -> dict:
        """
        Return the complete globals dictionary to pass to exec().
        Contains the five RLM functions, P, Q, and safe builtins.
        """
        scope = dict(_SAFE_BUILTINS)  # starts with safe builtins
        scope.update({
            # Document and question
            "P": self._document,
            "Q": self._question,
            # The five RLM functions
            "split":    self._fn_split,
            "peek":     self._fn_peek,
            "sub_call": self._fn_sub_call,
            "merge":    self._fn_merge,
            "final":    self._fn_final,
        })
        return scope


# ---------------------------------------------------------------------------
# REPLResult — what REPLExecutor.run() returns
# ---------------------------------------------------------------------------

@dataclass
class REPLResult:
    """Outcome of one REPLExecutor run (one document node)."""
    answer: str
    succeeded: bool
    turns_used: int
    llm_calls: int
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# REPLExecutor — the main REPL loop
# ---------------------------------------------------------------------------

class REPLExecutor:
    """
    Runs the REPL loop for a single Document node.

    Algorithm:
        for turn in range(max_turns):
            A. Build prompt  (preview + word count + fits flag + question + history)
            B. LLM generates Python code
            C. safe_exec(code) in REPLNamespace
            D. Append (code, result_or_error) to history
            E. If namespace._is_done -> return answer

        if max_turns exhausted -> return failure result

    Args:
        llm:       Any BaseLanguageModel instance.
        max_turns: Max REPL iterations before giving up (default 5).
        verbose:   Print per-turn debug info.
    """

    PREVIEW_WORDS = 150   # words shown to LLM as document preview
    EXEC_TIMEOUT_SEC = 1.0

    def __init__(
        self,
        llm,
        max_turns: int = 5,
        verbose: bool = False,
    ):
        self.llm = llm
        self.max_turns = max_turns
        self.verbose = verbose

    def run(
        self,
        document: Document,
        question: str,
        sub_call_handler: Callable[[Document, str], str],
    ) -> REPLResult:
        """
        Execute the REPL loop for *document* / *question*.

        Args:
            document:          The Document chunk to process.
            question:          The user's question.
            sub_call_handler:  Callable(doc, q) -> str — provided by RLMSystem.

        Returns:
            REPLResult with the answer (or failure info).
        """
        namespace = REPLNamespace(
            document=document,
            question=question,
            sub_call_handler=sub_call_handler,
        )
        exec_globals = namespace.as_exec_dict()

        history: List[str] = []
        llm_calls_start = self.llm.call_count

        for turn in range(1, self.max_turns + 1):
            # A. Build prompt
            prompt = self._build_prompt(
                document=document,
                question=question,
                history=history,
                turn=turn,
            )

            if self.verbose:
                print("  [REPL] Turn {}/{} | doc={!r} | fits={}".format(
                    turn, self.max_turns,
                    document.name[:40],
                    document.fits_in_window(self.llm.context_window),
                ))

            # B. Call LLM
            try:
                code = self.llm.generate(prompt)
            except Exception as e:
                err_msg = "LLM generation error: {}".format(e)
                if self.verbose:
                    print("  [REPL] " + err_msg)
                history.append("# Turn {} LLM error:\n# {}".format(turn, err_msg))
                continue

            code = _strip_markdown_fences(code)

            if self.verbose:
                print("  [REPL] Code ({} chars):\n{}".format(len(code), _indent(code, "    | ")))

            # C. Execute code safely
            error_msg = self._safe_exec(code, exec_globals)

            # D. Append to history
            if error_msg:
                history.append(
                    "# Turn {} code:\n{}\n# ERROR: {}".format(turn, code, error_msg)
                )
                if self.verbose:
                    print("  [REPL] Exec error: {}".format(error_msg))
            else:
                history.append("# Turn {} code (OK):\n{}".format(turn, code))

            # E. Check if done
            if namespace._is_done:
                llm_calls = self.llm.call_count - llm_calls_start
                if self.verbose:
                    print("  [REPL] Done after {} turn(s). Answer length: {} chars".format(
                        turn, len(namespace._answer)
                    ))
                return REPLResult(
                    answer=namespace._answer,
                    succeeded=True,
                    turns_used=turn,
                    llm_calls=llm_calls,
                )

        # Max turns exhausted
        llm_calls = self.llm.call_count - llm_calls_start
        reason = "REPL timeout: final() was never called after {} turns".format(self.max_turns)
        if self.verbose:
            print("  [REPL] " + reason)
        return REPLResult(
            answer="[timeout: no answer produced for '{}']".format(document.name),
            succeeded=False,
            turns_used=self.max_turns,
            llm_calls=llm_calls,
            failure_reason=reason,
        )

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        document: Document,
        question: str,
        history: List[str],
        turn: int,
    ) -> str:
        """
        Build the structured prompt shown to the LLM each REPL turn.

        Structure:
          DOCUMENT METADATA
          DOCUMENT PREVIEW (first 150 words)
          QUESTION
          HISTORY (previous turns)
          INSTRUCTIONS
        """
        fits = document.fits_in_window(self.llm.context_window)
        preview = document.preview(self.PREVIEW_WORDS)

        parts = [
            "=== RLM REPL Turn {}/{} ===".format(turn, self.max_turns),
            "",
            "DOCUMENT NAME: {}".format(document.name),
            "WORD COUNT: {:,}".format(document.word_count),
            "CONTEXT WINDOW: {:,} words".format(self.llm.context_window),
            "FITS IN WINDOW: {}".format(fits),
            "",
            "DOCUMENT PREVIEW (first {} words):".format(self.PREVIEW_WORDS),
            preview,
            "",
            "QUESTION: {}".format(question),
        ]

        if history:
            parts += [
                "",
                "=== PREVIOUS TURNS (for self-correction) ===",
            ]
            parts += history[-3:]  # show last 3 turns to avoid prompt bloat

        parts += [
            "",
            "=== YOUR TASK ===",
            "",
        ]

        if fits:
            parts += [
                "The document FITS in the context window.",
                "Read the preview above (it contains the full document text if word count <= {}).".format(
                    self.PREVIEW_WORDS
                ),
                "If the preview is truncated, use peek(P, 0, WORD_COUNT) to inspect the full chunk before answering.",
                "Answer directly only after you have seen enough of the chunk.",
                "You may still split if the chunk appears internally too large to reason about.",
                "",
                "Write Python code using only: split(P, k), peek(P, start, end), sub_call(chunk, Q), merge(results), final(answer).",
                "End with exactly one call to final().",
            ]
        else:
            k_suggestion = max(
                2,
                (document.word_count + self.llm.context_window - 1) // self.llm.context_window,
            )
            parts += [
                "The document does NOT fit in the context window.",
                "You MUST split it into chunks and call sub_call() on each chunk.",
                "Suggested split size: split(P, {})  -- adjust if needed.".format(k_suggestion),
                "",
                "Write Python code using only: split(P, k), peek(P, start, end), sub_call(chunk, Q), merge(results), final(answer).",
                "End with exactly one call to final().",
            ]

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Safe exec wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_exec(
        code: str,
        exec_globals: dict,
        timeout_sec: float = EXEC_TIMEOUT_SEC,
    ) -> Optional[str]:
        """
        Execute *code* in *exec_globals*.
        Returns None on success, or an error string on failure.
        """
        if not code.strip():
            return "Empty code block - nothing to execute."
        try:
            tree = ast.parse(code, filename="<llm_code>", mode="exec")
            _SafeCodeValidator().visit(tree)
            compiled = compile(tree, "<llm_code>", "exec")
            deadline = time.monotonic() + max(0.01, timeout_sec)

            def _trace(frame, event, arg):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "LLM code exceeded execution time limit ({:.2f}s).".format(timeout_sec)
                    )
                return _trace

            previous_trace = sys.gettrace()
            sys.settrace(_trace)
            try:
                exec(compiled, exec_globals)
            finally:
                sys.settrace(previous_trace)
            return None
        except Exception:
            return traceback.format_exc(limit=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """
    Remove ```python ... ``` or ``` ... ``` fences from LLM output.
    Returns the raw code inside (or the whole text if no fences found).
    """
    lines = text.strip().splitlines()
    # Strip opening fence
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # Strip closing fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _indent(text: str, prefix: str) -> str:
    """Add *prefix* to every line of *text*."""
    return "\n".join(prefix + line for line in text.splitlines())
