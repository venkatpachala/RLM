"""
Shared prompt builders for deterministic and REPL-driven RLM flows.
"""

from __future__ import annotations


def build_leaf_prompt(document_name: str, word_count: int, question: str, content: str) -> str:
    """Build the leaf-level analysis prompt used by optimized mode."""
    return (
        "DOCUMENT: {}\n"
        "WORD COUNT: {}\n"
        "QUESTION: {}\n\n"
        "CHUNK TEXT:\n{}\n\n"
        "Answer the question directly using only this chunk. "
        "If the question is asking for a specific fact, quote the exact fact if present. "
        "If this chunk does not contain enough evidence, say so briefly instead of guessing. "
        "If this is part of a larger document, return a concise chunk-level answer."
    ).format(
        document_name,
        word_count,
        question,
        content,
    )


def build_synthesis_prompt(document_name: str, question: str, partial_answers: list[str]) -> str:
    """Build the synthesis prompt that merges chunk-level answers."""
    return (
        "DOCUMENT: {}\n"
        "QUESTION: {}\n\n"
        "Synthesize the following chunk summaries into one accurate, concise, non-redundant answer.\n"
        "Prefer answers supported by multiple chunks or more specific evidence.\n"
        "If chunks disagree, keep the more precise answer and mention uncertainty only if necessary.\n\n"
        "{}"
    ).format(
        document_name,
        question,
        "\n\n".join("- " + answer for answer in partial_answers),
    )


def build_repl_task_suffix(fits: bool, preview_words: int, k_suggestion: int) -> list[str]:
    """Return the task section appended to each REPL prompt."""
    common = [
        "Use the live variable P for the document object. Never pass a filename string like 'document.pdf' into split() or peek().",
        "If you use keyword arguments, use exactly these names: split(P=..., k=...), peek(P=..., start=..., end=...), sub_call(doc=..., q=...).",
        "Core helpers available: split(P, k), peek(P, start, end), sub_call(chunk, Q), merge(results), final(answer).",
        "You may also use: find_all(text, needle), count_matches(text, needle), regex_search(pattern, text), parse_json(text), to_json(obj), FINAL(answer), FINAL_VAR(value).",
        "",
        "Write Python code using only the provided helpers and safe Python syntax.",
        "End with exactly one call to final() or FINAL().",
    ]
    if fits:
        return [
            "The document FITS in the context window.",
            "Read the preview above (it contains the full document text if word count <= {}).".format(
                preview_words
            ),
            "If the preview is truncated, use peek(P, 0, WORD_COUNT) to inspect the full chunk before answering.",
            "Answer directly only after you have seen enough of the chunk.",
            "You may still split if the chunk appears internally too large to reason about.",
            "",
        ] + common
    return [
        "The document does NOT fit in the context window.",
        "You MUST split it into chunks and call sub_call() on each chunk.",
        "Suggested split size: split(P, {})  -- adjust if needed.".format(k_suggestion),
        "",
    ] + common
