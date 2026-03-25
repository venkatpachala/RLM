"""
core/llm.py - Layer 2: The LLM Oracle
=======================================

Provides two LLM backends:

  OpenRouterLLM   -- Real NVIDIA Llama-3.1-Nemotron-70B via OpenRouter API.
                     Sends structured prompts, returns Python code strings.

  MockLLM         -- Offline simulation for testing (no API key required).
                     Generates deterministic, syntactically-correct Python
                     code that exercises the full RLM split/merge loop.

Both share the same BaseLanguageModel interface:
  .generate(prompt: str) -> str       # core oracle call
  .fits_in_window(text: str) -> bool  # True if text fits context window
  .call_count                         # how many LLM calls made so far
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLanguageModel(ABC):
    """Abstract base class for all LLM backends."""

    def __init__(self, context_window: int = 6000):
        self.context_window: int = context_window
        self.call_count: int = 0
        self._total_prompt_words: int = 0
        self._total_response_words: int = 0

    def fits_in_window(self, text: str) -> bool:
        """Return True when *text* word count fits inside the context window."""
        return len(text.split()) <= self.context_window

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the raw response string."""

    @abstractmethod
    def complete_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a normal text-generation prompt and return the response string."""

    def _record_call(self, prompt: str, response: str) -> None:
        self.call_count += 1
        self._total_prompt_words += len(prompt.split())
        self._total_response_words += len(response.split())

    def stats(self) -> dict:
        return {
            "call_count": self.call_count,
            "total_prompt_words": self._total_prompt_words,
            "total_response_words": self._total_response_words,
        }


# ---------------------------------------------------------------------------
# OpenRouter / NVIDIA Llama backend
# ---------------------------------------------------------------------------

OPENROUTER_SYSTEM_PROMPT = """\
You are a Python code generator for the RLM (Recursive Language Model) system.

You have access to these pre-defined Python functions. Call them EXACTLY as shown:

  split(P, k)        -- Split document P into k equal chunks.
                        Returns a list of Document objects.

  sub_call(doc, q)   -- Recursively process document chunk `doc` with question `q`.
                        Returns a string answer.

  merge(answers)     -- Merge a list of string answers into one combined answer.
                        Filters blanks and joins non-empty ones.

  final(answer)      -- Declare your final answer. This ENDS the current REPL turn.
                        You MUST call final() at the end of every code block.

  peek(P, start, end)-- Read words[start:end] of document P as a string.

Rules you MUST follow:
  1. Output ONLY valid Python code — no markdown, no backticks, no explanation.
  2. Always call final() exactly once, at the end.
  3. Never import anything. All functions are already in scope.
  4. When the document fits in the window (fits=True), use peek() if you need more than the preview.
     Answer directly only after you have inspected enough text.
  5. When the document does NOT fit (fits=False), call split() then sub_call() on each chunk.
  6. Use merge() to combine sub_call results before calling final().
  7. Use the live variables exactly as provided:
     - Use P for the current Document object, never a string like "document.pdf"
     - Use Q for the current question unless you intentionally rewrite it
  8. Valid examples:
     - chunks = split(P, 4)
     - results.append(sub_call(doc=chunk, q=Q))
  9. Invalid examples:
     - split(P="document.pdf", k=4)
     - split("document.pdf", 4)
     - sub_call(chunk=chunk, question=Q)

Example (document fits in window):
  final("The document discusses climate change impacts on coastal regions.")

Example (document does NOT fit, split into 4):
  chunks = split(P, 4)
  results = [sub_call(c, Q) for c in chunks]
  combined = merge(results)
final(combined)
"""

TEXT_SYSTEM_PROMPT = """\
You are a precise document-analysis assistant.

Rules:
1. Answer in normal text, not code.
2. Be concise, factual, and non-redundant.
3. If asked to synthesize chunk summaries, merge them into one coherent answer.
4. Do not mention missing context unless it is genuinely necessary.
"""


class OpenRouterLLM(BaseLanguageModel):
    """
    Calls NVIDIA Llama-3.1-Nemotron-70B via the OpenRouter REST API.

    Args:
        api_key:        Your OpenRouter API key (sk-or-v1-...).
        context_window: Max words per chunk sent to LLM (default 6000).
        model:          OpenRouter model slug to use.
        verbose:        Print call start/end markers.
    """

    MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        context_window: int = 6000,
        model: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(context_window=context_window)
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Get a free key at https://openrouter.ai/keys"
            )
        if not HAS_REQUESTS:
            raise ImportError(
                "The 'requests' library is required.\n"
                "Install it with:  pip install requests"
            )
        self.api_key = api_key
        self.model = model or self.MODEL
        self.verbose = verbose

    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to OpenRouter and return the response text.
        Raises RuntimeError on API errors after retries.
        """
        if self.verbose:
            print("  [LLM #{} -> {}]".format(self.call_count + 1, self.model))

        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rlm-project/rlm_v0",
            "X-Title": "RLM Document Processor",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": OPENROUTER_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = _requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                response_text = data["choices"][0]["message"]["content"].strip()
                self._record_call(prompt, response_text)
                return response_text

            except _requests.exceptions.Timeout:
                last_error = TimeoutError("OpenRouter API timed out (60s)")
                if attempt < 2:
                    time.sleep(2 ** attempt)

            except _requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                try:
                    body = e.response.json()
                    msg = body.get("error", {}).get("message", str(e))
                except Exception:
                    msg = str(e)
                last_error = RuntimeError(
                    "OpenRouter HTTP {} error: {}".format(status, msg)
                )
                if status in (429, 500, 502, 503) and attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    break

            except Exception as e:
                last_error = e
                break

        raise RuntimeError(
            "OpenRouter API call failed after retries: {}".format(last_error)
        )

    def complete_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a standard text prompt to OpenRouter and return plain text."""
        if self.verbose:
            print("  [LLM #{} -> {} | text]".format(self.call_count + 1, self.model))

        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rlm-project/rlm_v0",
            "X-Title": "RLM Document Processor",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or TEXT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = _requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                response_text = data["choices"][0]["message"]["content"].strip()
                self._record_call(prompt, response_text)
                return response_text
            except _requests.exceptions.Timeout:
                last_error = TimeoutError("OpenRouter API timed out (60s)")
                if attempt < 2:
                    time.sleep(2 ** attempt)
            except _requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                try:
                    body = e.response.json()
                    msg = body.get("error", {}).get("message", str(e))
                except Exception:
                    msg = str(e)
                last_error = RuntimeError(
                    "OpenRouter HTTP {} error: {}".format(status, msg)
                )
                if status in (429, 500, 502, 503) and attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    break
            except Exception as e:
                last_error = e
                break

        raise RuntimeError(
            "OpenRouter text call failed after retries: {}".format(last_error)
        )


# ---------------------------------------------------------------------------
# MockLLM — for offline testing
# ---------------------------------------------------------------------------

class MockLLM(BaseLanguageModel):
    """
    Offline mock that generates deterministic RLM-compatible Python code.

    Modes:
      'smart'        -- Always writes correct split/sub_call/merge/final code.
      'direct'       -- If doc fits, writes a direct final() answer.
                        (Used automatically when doc fits in window.)

    Args:
        context_window: Max words per chunk (default 6000).
        mode:           'smart' (default).
        verbose:        Print simulated responses.
    """

    def __init__(
        self,
        context_window: int = 6000,
        mode: str = "smart",
        verbose: bool = False,
    ):
        super().__init__(context_window=context_window)
        self.mode = mode
        self.verbose = verbose

    def generate(self, prompt: str) -> str:
        """
        Inspect the prompt for context clues, then return synthetic Python code.
        The generated code always correctly calls final().
        """
        response = self._build_response(prompt)
        if self.verbose:
            print("  [MockLLM #{}] Generated {} words of code".format(
                self.call_count + 1, len(response.split())
            ))
        self._record_call(prompt, response)
        return response

    def complete_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Return a deterministic text answer for testing the optimized mode."""
        response = self._build_text_response(prompt)
        self._record_call(prompt, response)
        return response

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    def _build_response(self, prompt: str) -> str:
        """
        Parse the prompt to determine fits/no-fits, then return the
        appropriate code pattern.
        """
        fits = self._prompt_says_fits(prompt)
        word_count = self._extract_word_count(prompt)

        if fits:
            # Document fits in window: answer directly
            return (
                'final("This section covers information extracted directly '
                'from the document chunk ({} words).  '
                '[MockLLM simulated answer]")'.format(word_count)
            )
        else:
            # Document does NOT fit: split into optimal k, recurse
            k = self._optimal_k(word_count)
            return (
                "chunks = split(P, {k})\n"
                "results = [sub_call(c, Q) for c in chunks]\n"
                "combined = merge(results)\n"
                "final(combined)"
            ).format(k=k)

    def _prompt_says_fits(self, prompt: str) -> bool:
        """
        Check whether the REPL prompt indicates the document fits in the window.
        The REPLExecutor injects 'FITS IN WINDOW: True/False' into the prompt.
        """
        lower = prompt.lower()
        if "fits in window: true" in lower:
            return True
        if "fits in window: false" in lower:
            return False
        # Fallback: estimate from the word count line
        wc = self._extract_word_count(prompt)
        return wc <= self.context_window

    def _extract_word_count(self, prompt: str) -> int:
        """
        Extract 'WORD COUNT: N' from the prompt, or fall back to 0.
        """
        for line in prompt.splitlines():
            lower = line.lower()
            if "word count:" in lower:
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        return int(parts[-1].strip().replace(",", ""))
                    except ValueError:
                        pass
        return 0

    def _optimal_k(self, word_count: int) -> int:
        """
        Choose the number of chunks so each chunk fits in the context window.
        Minimum 2, safety maximum 20.
        """
        if word_count <= 0 or self.context_window <= 0:
            return 2
        k = (word_count + self.context_window - 1) // self.context_window
        return max(2, min(k, 20))

    def _build_text_response(self, prompt: str) -> str:
        if "Synthesize the following chunk summaries" in prompt:
            return "Unified summary based on chunk-level answers."
        for line in prompt.splitlines():
            if line.startswith("QUESTION:"):
                question = line.split(":", 1)[1].strip()
                return "Mock answer for: {}".format(question)
        return "Mock answer."
