import unittest

from core.document import Document
from core.llm import BaseLanguageModel, MockLLM
from core.repl import REPLExecutor, REPLNamespace
from core.rlm_system import RLMSystem


class DirectAnswerLLM(BaseLanguageModel):
    def __init__(self, answer: str, context_window: int = 6000):
        super().__init__(context_window=context_window)
        self.answer = answer

    def generate(self, prompt: str) -> str:
        response = 'final("{}")'.format(self.answer.replace('"', '\\"'))
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


if __name__ == "__main__":
    unittest.main()
