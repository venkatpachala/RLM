import json
import tempfile
import unittest

from core.document import Document
from core.llm import BaseLanguageModel, MockLLM
from core.optimized_rlm import OptimizedRLMConfig, OptimizedRLMSystem
from core.repl import REPLExecutor, REPLNamespace, _strip_markdown_fences
from core.rlm_system import RLMSystem


class DirectAnswerLLM(BaseLanguageModel):
    def __init__(self, answer: str, context_window: int = 6000):
        super().__init__(context_window=context_window)
        self.answer = answer

    def generate(self, prompt: str) -> str:
        response = 'final("{}")'.format(self.answer.replace('"', '\\"'))
        self._record_call(prompt, response)
        return response

    def complete_text(self, prompt: str, system_prompt=None) -> str:
        response = self.answer
        self._record_call(prompt, response)
        return response


class AlwaysSplitLLM(BaseLanguageModel):
    def __init__(self, context_window: int = 1):
        super().__init__(context_window=context_window)

    def generate(self, prompt: str) -> str:
        response = (
            "chunks = split(P, 2)\n"
            "results = [sub_call(c, Q) for c in chunks]\n"
            "final(merge(results))"
        )
        self._record_call(prompt, response)
        return response

    def complete_text(self, prompt: str, system_prompt=None) -> str:
        response = "split summary"
        self._record_call(prompt, response)
        return response


class SynthesisLLM(BaseLanguageModel):
    def __init__(self, context_window: int = 6000):
        super().__init__(context_window=context_window)

    def generate(self, prompt: str) -> str:
        if "Synthesize these chunk-level answers" in prompt:
            response = 'final("Unified summary.")'
        else:
            response = (
                "chunks = split(P, 2)\n"
                "results = [sub_call(c, Q) for c in chunks]\n"
                "final(merge(results))"
            )
        self._record_call(prompt, response)
        return response

    def complete_text(self, prompt: str, system_prompt=None) -> str:
        response = "Unified summary."
        self._record_call(prompt, response)
        return response


class TextOnlyLLM(BaseLanguageModel):
    def __init__(self, context_window: int = 6000):
        super().__init__(context_window=context_window)

    def generate(self, prompt: str) -> str:
        response = 'final("unused")'
        self._record_call(prompt, response)
        return response

    def complete_text(self, prompt: str, system_prompt=None) -> str:
        if "Synthesize the following chunk summaries" in prompt:
            response = "Final synthesized answer."
        else:
            response = "Leaf summary."
        self._record_call(prompt, response)
        return response


class TokenLimitLLM(BaseLanguageModel):
    def __init__(self, context_window: int = 6000):
        super().__init__(context_window=context_window)

    def generate(self, prompt: str) -> str:
        response = 'final("unused")'
        self._record_call(prompt, response)
        return response

    def complete_text(self, prompt: str, system_prompt=None) -> str:
        if "Synthesize the following chunk summaries" in prompt:
            response = "Backoff synthesis."
            self._record_call(prompt, response)
            return response
        if "WORD COUNT: 4" in prompt:
            raise RuntimeError("OpenRouter text call failed after retries: Prompt tokens limit exceeded: 4889 > 1954")
        response = "Recovered leaf answer."
        self._record_call(prompt, response)
        return response


class RLMTests(unittest.TestCase):
    def test_prompt_mentions_peek_for_fits_document(self):
        executor = REPLExecutor(llm=MockLLM(context_window=1000), max_turns=2)
        doc = Document(name="doc", content="word " * 200)

        prompt = executor._build_prompt(doc, "Summarize", history=[], turn=1)

        self.assertIn("peek(P, 0, WORD_COUNT)", prompt)
        self.assertIn("peek(P, start, end)", prompt)

    def test_safe_exec_blocks_attribute_escape(self):
        doc = Document(name="doc", content="hello world")
        namespace = REPLNamespace(doc, "Q", lambda child, q: "ok")
        code = "final(str(split.__func__.__globals__))"

        error = REPLExecutor._safe_exec(code, namespace.as_exec_dict())

        self.assertIsNotNone(error)
        self.assertIn("Attribute access is not allowed", error)

    def test_safe_exec_allows_list_append_pattern(self):
        doc = Document(name="doc", content="hello world")
        namespace = REPLNamespace(doc, "Q", lambda child, q: "ok")
        code = (
            "results = []\n"
            "results.append('ok')\n"
            "final(merge(results))"
        )

        error = REPLExecutor._safe_exec(code, namespace.as_exec_dict())

        self.assertIsNone(error)
        self.assertEqual(namespace._answer, "ok")

    def test_repl_functions_accept_prompt_keyword_names(self):
        doc = Document(name="doc", content="one two three four")
        namespace = REPLNamespace(doc, "Q", lambda child, q: q)
        code = (
            "chunks = split(P=P, k=2)\n"
            "results = []\n"
            "for chunk in chunks:\n"
            "    results.append(sub_call(doc=chunk, q=Q))\n"
            "final(merge(results))"
        )

        error = REPLExecutor._safe_exec(code, namespace.as_exec_dict())

        self.assertIsNone(error)
        self.assertEqual(namespace._answer, "Q\n\nQ")

    def test_safe_exec_times_out_busy_code(self):
        doc = Document(name="doc", content="hello world")
        namespace = REPLNamespace(doc, "Q", lambda child, q: "ok")
        code = "for i in range(100000000):\n    pass"

        error = REPLExecutor._safe_exec(
            code,
            namespace.as_exec_dict(),
            timeout_sec=0.01,
        )

        self.assertIsNotNone(error)
        self.assertIn("execution time limit", error)

    def test_safe_exec_allows_string_augassign(self):
        doc = Document(name="doc", content="hello world")
        namespace = REPLNamespace(doc, "Q", lambda child, q: "ok")
        code = 'answer = "A"\nanswer += "B"\nfinal(answer)'

        error = REPLExecutor._safe_exec(code, namespace.as_exec_dict())

        self.assertIsNone(error)
        self.assertEqual(namespace._answer, "AB")

    def test_strip_markdown_fences_extracts_code_after_prose(self):
        text = (
            "Since the document fits, we can answer directly.\n\n"
            "```python\n"
            "final(\"hello\")\n"
            "```"
        )

        code = _strip_markdown_fences(text)

        self.assertEqual(code, 'final("hello")')

    def test_run_success_is_not_based_on_answer_text(self):
        llm = DirectAnswerLLM("There is no answer in the appendix.")
        doc = Document(name="doc", content="short content")

        result = RLMSystem(llm=llm).run(doc, "What does it say?")

        self.assertTrue(result.succeeded)
        self.assertIsNone(result.failure_reason)
        self.assertIn("no answer", result.answer.lower())

    def test_depth_limit_marks_whole_run_failed(self):
        llm = AlwaysSplitLLM(context_window=1)
        doc = Document(name="doc", content="one two three four")

        result = RLMSystem(llm=llm, max_depth=0, max_turns_per_node=2).run(
            doc,
            "Summarize",
        )

        self.assertFalse(result.succeeded)
        self.assertIn("depth limit", result.failure_reason.lower())

    def test_run_synthesizes_multi_paragraph_answers(self):
        llm = SynthesisLLM()
        system = RLMSystem(llm=llm, max_depth=2, max_turns_per_node=2)
        doc = Document(name="doc", content="one two three four")

        result = system._synthesize_final_answer(
            doc,
            "Summarize",
            "Point A.\n\nPoint B.\n\nPoint C.",
        )

        self.assertEqual(result, "Unified summary.")

    def test_optimized_rlm_runs_and_synthesizes(self):
        llm = TextOnlyLLM()
        config = OptimizedRLMConfig(leaf_chunk_words=3, chunk_overlap_words=0, max_llm_calls=20)
        doc = Document(name="doc", content="one two three four five six")

        result = OptimizedRLMSystem(llm=llm, config=config).run(doc, "Summarize")

        self.assertTrue(result.succeeded)
        self.assertEqual(result.answer, "Final synthesized answer.")

    def test_optimized_rlm_writes_jsonl_logs(self):
        llm = TextOnlyLLM()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = tmpdir + "/trajectory.jsonl"
            config = OptimizedRLMConfig(
                leaf_chunk_words=10,
                log_path=log_path,
                max_llm_calls=10,
            )
            doc = Document(name="doc", content="one two three four")

            result = OptimizedRLMSystem(llm=llm, config=config).run(doc, "Summarize")

            self.assertTrue(result.succeeded)
            with open(log_path, "r", encoding="utf-8") as fh:
                events = [json.loads(line)["event"] for line in fh if line.strip()]
            self.assertIn("run_start", events)
            self.assertIn("run_end", events)

    def test_optimized_rlm_backs_off_after_prompt_limit_error(self):
        llm = TokenLimitLLM()
        config = OptimizedRLMConfig(
            leaf_chunk_words=4,
            min_leaf_chunk_words=2,
            chunk_overlap_words=0,
            max_llm_calls=20,
        )
        doc = Document(name="doc", content="one two three four")

        result = OptimizedRLMSystem(llm=llm, config=config).run(doc, "Summarize")

        self.assertTrue(result.succeeded)
        self.assertEqual(result.answer, "Backoff synthesis.")

    def test_lookup_question_uses_focused_excerpts(self):
        llm = TextOnlyLLM()
        config = OptimizedRLMConfig(leaf_chunk_words=100, max_llm_calls=10)
        system = OptimizedRLMSystem(llm=llm, config=config)
        doc = Document(
            name="doc",
            content=(
                "Alpha beta gamma. Dr. Sagarika Dash can be contacted at "
                "sagarika.dash@example.com for parliamentary matters. "
                "Additional unrelated text follows."
            ),
        )

        selected = system._select_leaf_content(doc, "What is the E-mail of Dr. Sagarika Dash?")

        self.assertIn("EXCERPT 1", selected)
        self.assertIn("sagarika.dash@example.com", selected)


if __name__ == "__main__":
    unittest.main()
