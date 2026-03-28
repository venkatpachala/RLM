import re
import time
import random
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional



@dataclass
class LLMConfig:
    name: str
    context_window: int     # K: max words the model handles reliably
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0



class BaseLLM(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.call_count = 0
        self.total_input_words = 0
        self.total_output_words = 0

    @property
    def context_window(self) -> int:
        return self.config.context_window

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def complete_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Text-completion style helper used by the optimized pipeline.
        Providers can override this for a cleaner non-code path.
        """
        return self.generate(prompt)

    def _record(self, prompt: str, response: str):
        self.call_count += 1
        self.total_input_words += len(prompt.split())
        self.total_output_words += len(response.split())

    def stats(self) -> str:
        return (f"LLM '{self.config.name}' | "
                f"Calls: {self.call_count} | "
                f"Input words: {self.total_input_words:,} | "
                f"Output words: {self.total_output_words:,}")


class OpenRouterLLM(BaseLLM):
    """
    Calls NVIDIA's Llama-3.1 Nemotron 70B via OpenRouter.
    This model is free to use with an OpenRouter API key.

    Model: nvidia/llama-3.1-nemotron-70b-instruct
    Why this model:
      - 70B parameters → strong reasoning + code writing
      - Free tier on OpenRouter
      - 131K context window → can handle large document chunks
      - Good instruction following (critical for RLM code generation)

    Get your free API key at: https://openrouter.ai/keys
    """

    MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    SYSTEM_PROMPT = """You are an RLM (Recursive Language Model) agent.
    You are solving a reasoning task over a long document.
    The document is stored externally as variable P — you CANNOT read it all at once.

    Your job: write Python code using only these functions:

    split(P, k)                → splits document P into k equal chunks (list of Documents)
    peek(P, start, end)        → reads words[start:end] from P as a string (cheap, no cost)
    sub_call(chunk, question)  → recursively solves question on a smaller chunk (returns string)
    merge(list_of_answers)     → combines a list of string answers into one string
    final(answer_string)       → SIGNALS COMPLETION — you MUST call this when you have the answer

    RULES:
    1. Always end with final("your answer here") — this is mandatory
    2. If the document is too big, use split() and sub_call()
    3. If the document fits (DOCUMENT WORDS ≤ CONTEXT WINDOW), read and answer directly
    4. Write only clean Python — no markdown fences, no explanations
    5. sub_call() returns a plain string answer, not a Document
    6. Keep it simple — 5-10 lines of code maximum

    EXAMPLE for a document that fits:
    answer = "The document discusses..."
    final(answer)

    EXAMPLE for a large document:
    chunks = split(P, 4)
    results = [sub_call(c, question) for c in chunks]
    final(merge(results))
    """

    def __init__(self, api_key: str, context_window: int = 8000):
        config = LLMConfig(
            name="nvidia/llama-3.1-nemotron-70b-instruct",
            context_window=context_window,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )
        super().__init__(config)
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        """
        Call OpenRouter API with the NVIDIA model.
        Includes retry logic for rate limits and transient errors.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rlm-impl",
            "X-Title": "RLM Implementation",
        }

        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 800,
            "temperature": 0.1,   
        }

        for attempt in range(3):
            try:
                print(f"  [OpenRouter] Calling {self.MODEL}...")
                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  [OpenRouter] Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    error_body = response.text[:300]
                    raise RuntimeError(
                        f"OpenRouter API error {response.status_code}: {error_body}"
                    )

                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()

                # Strip markdown fences the model sometimes adds
                text = self._clean_code(text)

                print(f"  [OpenRouter] Response ({len(text.split())} words):\n"
                      f"  {text[:200]}{'...' if len(text) > 200 else ''}")

                self._record(prompt, text)
                return text

            except requests.Timeout:
                print(f"  [OpenRouter] Timeout on attempt {attempt+1}")
                if attempt == 2:
                    return 'final("Error: API timeout after 3 attempts.")'
                time.sleep(2)
            except Exception as e:
                if attempt == 2:
                    return f'final("Error calling API: {str(e)[:100]}")'
                time.sleep(1)

        return 'final("Error: all retry attempts failed.")'

    def complete_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Plain text completion for the deterministic optimized pipeline.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rlm-impl",
            "X-Title": "RLM Implementation",
        }

        payload = {
            "model": self.MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "Answer accurately and concisely using only the supplied document text.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 800,
            "temperature": 0.1,
        }

        for attempt in range(3):
            try:
                print(f"  [OpenRouter] Completing with {self.MODEL}...")
                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  [OpenRouter] Rate limited - waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if response.status_code != 200:
                    raise RuntimeError(
                        f"OpenRouter API error {response.status_code}: {response.text[:300]}"
                    )

                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
                self._record(prompt, text)
                return text
            except requests.Timeout:
                print(f"  [OpenRouter] Timeout on attempt {attempt + 1}")
                if attempt == 2:
                    raise RuntimeError("OpenRouter text call timed out after 3 attempts")
                time.sleep(2)
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"OpenRouter text call failed after retries: {exc}")
                time.sleep(1)

    def _clean_code(self, text: str) -> str:
        """Remove markdown fences if the model wrapped its code."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text


class OllamaLLM(BaseLLM):
    """
    Local Ollama-backed LLM. Defaults to a small Qwen model so the project can
    run without an OpenRouter key when Ollama is installed locally.
    """

    MODEL = "qwen2.5:3b"
    API_URL = "http://127.0.0.1:11434/api/generate"

    SYSTEM_PROMPT = OpenRouterLLM.SYSTEM_PROMPT

    def __init__(
        self,
        model: str = MODEL,
        context_window: int = 8000,
        api_url: str = API_URL,
    ):
        config = LLMConfig(
            name=f"ollama/{model}",
            context_window=context_window,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
        )
        super().__init__(config)
        self.model = model
        self.api_url = api_url

    def generate(self, prompt: str) -> str:
        full_prompt = (
            self.SYSTEM_PROMPT.strip()
            + "\n\nUSER TASK:\n"
            + prompt.strip()
        )
        text = self._request(full_prompt, temperature=0.1)
        cleaned = self._clean_code(text)
        self._record(prompt, cleaned)
        return cleaned

    def complete_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = (
            (system_prompt or "Answer accurately and concisely using only the supplied document text.").strip()
            + "\n\nUSER TASK:\n"
            + prompt.strip()
        )
        text = self._request(full_prompt, temperature=0.1)
        self._record(prompt, text)
        return text.strip()

    def _request(self, prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            print(f"  [Ollama] Calling {self.model}...")
            response = requests.post(self.api_url, json=payload, timeout=120)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error {response.status_code}: {response.text[:300]}"
                )
            data = response.json()
            return data.get("response", "").strip()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not reach Ollama at {}. Start Ollama and run 'ollama pull {}' first.".format(
                    self.api_url, self.model
                )
            ) from exc

    @staticmethod
    def _clean_code(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text



class MockLLM(BaseLLM):
    """
    Deterministic mock for testing without an API key.

    Simulates two behaviours:
      mode="smart"  — writes clean code, finds answers correctly
      mode="buggy"  — demonstrates the 4 RLM failure modes
    """

    def __init__(self, context_window: int = 500, mode: str = "smart",
                 verbose: bool = True):
        config = LLMConfig(name=f"mock-{mode}", context_window=context_window)
        super().__init__(config)
        self.mode = mode
        self.verbose = verbose

    def generate(self, prompt: str) -> str:
        doc_len = self._extract_doc_length(prompt)
        is_leaf = doc_len is None or doc_len <= self.config.context_window

        if self.verbose:
            leaf_str = "LEAF" if is_leaf else f"DECOMPOSE (doc={doc_len})"
            print(f"  [MockLLM #{self.call_count+1}] {leaf_str}")

        if self.mode == "buggy" and random.random() < 0.5:
            response = self._buggy(prompt)
        elif is_leaf:
            response = self._answer_leaf(prompt)
        else:
            response = self._decompose(doc_len)

        self._record(prompt, response)
        return response

    def complete_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self._answer_leaf(prompt)
        match = re.fullmatch(r'final\("(.*)"\)', response)
        text = match.group(1) if match else response
        self._record(prompt, text)
        return text

    def _decompose(self, doc_len: int) -> str:
        k = min(6, max(2, doc_len // self.config.context_window + 1))
        return f"""chunks = split(P, {k})
        results = [sub_call(c, question) for c in chunks]
        final(merge(results))"""

    def _answer_leaf(self, prompt: str) -> str:
        p = prompt.lower()
        if "penalty" in p:
            if "2%" in p or "section 14" in p or "late" in p:
                return 'final("FOUND: Section 14.3 — Late delivery penalty: 2%/week, capped at 20%.")'
            elif "section 22" in p or "ip breach" in p:
                return 'final("FOUND: Section 22.1 — IP breach: full contract value + legal fees.")'
            return 'final("No penalty clauses found in this section.")'
        return 'final("Section processed — no specific findings.")'

    def _buggy(self, prompt: str) -> str:
        bugs = [
            'chuncks = split(P, 3)\nresults = [sub_call(c, question) for c in chuncks]\nfinal(merge(results))',
            'result = sub_call(P, question)\nfinal(result)',   # infinite recursion
            'chunks = split(P, 2)\nfor c in chunks:\n    sub_call(c, question)\n# no final()',
        ]
        chosen = random.choice(bugs)
        print(f"  [MockLLM] SIMULATED BUG")
        return chosen

    def _extract_doc_length(self, prompt: str) -> Optional[int]:
        m = re.search(r'DOCUMENT WORDS:\s*(\d[\d,]*)', prompt, re.IGNORECASE)
        return int(m.group(1).replace(",", "")) if m else None


OpenAICompatibleLLM = OpenRouterLLM
