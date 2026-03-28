"""
main.py - RLM on your documents
================================

USAGE:

  Local default (Ollama + Qwen):
    python main.py --question "Summarize this document"

  OpenRouter:
    python main.py --key "sk-or-v1-..." --question "Summarize this document"

DEFAULT LOCAL MODEL:
  Ollama with qwen2.5:3b

GET A FREE API KEY FOR OPENROUTER:
  https://openrouter.ai/keys

INSTALL DEPENDENCIES FIRST:
  pip install pymupdf requests
  ollama pull qwen2.5:3b
"""

import os
import sys
import argparse
import traceback

# ============================================================
# EDIT THIS LINE — paste your OpenRouter key here directly
# ============================================================
API_KEY = ""   # e.g. "sk-or-v1-abc123..."
# ============================================================

CONTEXT_WINDOW_WORDS = 6000   # words per chunk sent to LLM


def main():
    # ── Parse command-line arguments ────────────────────────────────────────
    parser = argparse.ArgumentParser(description="RLM - Recursive Language Model")
    parser.add_argument("--key",      type=str, default="", help="OpenRouter API key")
    parser.add_argument("--question", type=str, default="", help="Question to ask")
    parser.add_argument("--folder",   type=str, default="", help="Document folder path")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434/api/generate", help="Ollama generate API URL")
    parser.add_argument("--ollama-timeout-seconds", type=int, default=600, help="Per-request timeout for Ollama calls")
    parser.add_argument(
        "--mode",
        type=str,
        default="optimized",
        choices=("optimized", "repl"),
        help="Execution mode: deterministic optimized RLM or REPL-driven RLM",
    )
    parser.add_argument("--log", type=str, default="", help="Optional JSONL trajectory log path")
    parser.add_argument("--max-llm-calls", type=int, default=100, help="LLM call budget for optimized mode")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="Time budget for optimized mode")
    parser.add_argument("--leaf-chunk-words", type=int, default=800, help="Target leaf chunk size for optimized mode")
    parser.add_argument("--min-leaf-chunk-words", type=int, default=200, help="Minimum leaf chunk size after adaptive backoff")
    args = parser.parse_args()

    # Resolve folder: use arg if given, else default to <script_dir>/document_pdf
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.folder:
        doc_folder = args.folder
    else:
        doc_folder = os.path.join(script_dir, "document_pdf")

    print("=" * 60)
    print("  RLM - Recursive Language Model")
    print("=" * 60)
    print()

    # ── Step 1: Imports ──────────────────────────────────────────────────────
    try:
        sys.path.insert(0, script_dir)
        from core.document import load_document_from_folder
    except ImportError as e:
        print("IMPORT ERROR: {}".format(e))
        print("Fix: make sure you are running python from inside the rlm_real/ folder")
        sys.exit(1)

    # ── Step 2: Load document ────────────────────────────────────────────────
    print("[Step 1/4] Loading document from '{}' folder...".format(doc_folder))
    try:
        document = load_document_from_folder(doc_folder)
    except FileNotFoundError as e:
        print()
        print("ERROR: " + str(e))
        print()
        print("FIX: Drop a PDF or .txt file into the document_pdf/ folder")
        sys.exit(1)
    except Exception as e:
        print("ERROR loading document: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    print("  Name    : {}".format(document.name))
    print("  Words   : {:,}".format(len(document)))
    print("  Preview : {}".format(document.preview(20)))
    print()

    # ── Step 3: Resolve API key ──────────────────────────────────────────────
    # Priority: --key arg > hardcoded API_KEY > environment variable
    resolved_key = (
        args.key.strip()
        or API_KEY.strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    print("[Step 2/4] Setting up LLM...")

    if resolved_key:
        print("  Provider: OpenRouter")
        print("  Model   : nvidia/llama-3.1-nemotron-70b-instruct")
        print("  Window  : {:,} words per chunk".format(CONTEXT_WINDOW_WORDS))
        from core.llm import OpenRouterLLM
        llm = OpenRouterLLM(api_key=resolved_key, context_window=CONTEXT_WINDOW_WORDS)
    else:
        print("  Provider: Ollama")
        print("  Model   : {}".format(args.ollama_model))
        print("  Endpoint: {}".format(args.ollama_url))
        print("  Window  : {:,} words per chunk".format(CONTEXT_WINDOW_WORDS))
        print("  Tip     : Run 'ollama pull {}' first if needed".format(args.ollama_model))
        from core.llm import OllamaLLM
        llm = OllamaLLM(
            model=args.ollama_model,
            context_window=CONTEXT_WINDOW_WORDS,
            api_url=args.ollama_url,
            request_timeout_sec=args.ollama_timeout_seconds,
        )

    print()

    # ── Step 5: Get question ─────────────────────────────────────────────────
    print("[Step 3/4] Question setup...")

    if args.question.strip():
        # Question passed as argument — no need to prompt
        question = args.question.strip()
        print("  Question (from --question arg): {}".format(question))
    else:
        # Ask interactively
        print("  What would you like to know about this document?")
        print()
        print("  Example questions:")
        print("    - Summarize the main points")
        print("    - Find all dates and deadlines mentioned")
        print("    - What are the key topics covered?")
        print("    - Extract all important numbers and figures")
        print()
        try:
            sys.stdout.write("  Your question: ")
            sys.stdout.flush()
            question = sys.stdin.readline().strip()
        except (EOFError, KeyboardInterrupt):
            question = ""

        if not question:
            question = "Summarize the main points of this document."
            print("  (No question entered — using default)")

        print("  Using: {}".format(question))

    print()

    # ── Step 6: Run RLM ──────────────────────────────────────────────────────
    print("[Step 4/4] Running RLM...")
    print()
    print("  Mode:")
    if args.mode == "optimized":
        print("  - Optimized deterministic recursion with LLM summarization/synthesis")
    else:
        print("  - REPL-driven recursive code generation")
    print("  What will happen:")
    if document.fits_in_window(CONTEXT_WINDOW_WORDS):
        print("  - Document fits in one window -> single LLM call")
    else:
        chunks_needed = max(2, len(document) // CONTEXT_WINDOW_WORDS + 1)
        print("  - Document is {:,} words, window is {:,}".format(
            len(document), CONTEXT_WINDOW_WORDS))
        print("  - Will split into ~{} chunks recursively".format(chunks_needed))
        print("  - Each chunk answered separately, then merged")
    print()

    try:
        if args.mode == "optimized":
            from core.optimized_rlm import OptimizedRLMConfig, OptimizedRLMSystem
            config = OptimizedRLMConfig(
                max_depth=6,
                leaf_chunk_words=min(args.leaf_chunk_words, CONTEXT_WINDOW_WORDS),
                min_leaf_chunk_words=max(50, min(args.min_leaf_chunk_words, args.leaf_chunk_words)),
                chunk_overlap_words=150,
                max_llm_calls=args.max_llm_calls,
                max_elapsed_sec=args.max_seconds,
                enable_final_synthesis=True,
                log_path=args.log or None,
            )
            rlm = OptimizedRLMSystem(llm=llm, config=config, verbose=True)
        else:
            from core.rlm_system import RLMSystem
            rlm = RLMSystem(llm=llm, max_depth=6, max_turns_per_node=5, verbose=True)
        result = rlm.run(document, question)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print("\nERROR during RLM execution: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    # ── Step 7: Save result ───────────────────────────────────────────────────
    output_path = os.path.join(script_dir, "rlm_result.txt")
    repl_output_path = os.path.join(script_dir, "rlm_result.py")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("RLM Result\n")
            f.write("=" * 60 + "\n")
            f.write("Document : {}\n".format(document.name))
            f.write("Question : {}\n".format(question))
            f.write("LLM calls: {}\n".format(result.llm_calls))
            f.write("Time     : {:.1f}s\n".format(result.elapsed_sec))
            f.write("Success  : {}\n".format(result.succeeded))
            f.write("\nAnswer:\n")
            f.write("-" * 60 + "\n")
            f.write(result.answer + "\n")
            f.write("-" * 60 + "\n")
        print("\nResult saved to: {}".format(output_path))

        if args.mode == "repl" and getattr(result, "repl_trace", None):
            with open(repl_output_path, "w", encoding="utf-8") as f:
                f.write("# REPL trace generated by the RLM run\n")
                f.write("# Document : {}\n".format(document.name))
                f.write("# Question : {}\n".format(question))
                f.write("# Generated Python code by node/turn follows.\n\n")
                f.write("\n\n".join(result.repl_trace))
                f.write("\n")
            print("REPL trace saved to: {}".format(repl_output_path))
    except Exception as e:
        print("\nCould not save result file: {}".format(e))
        print("\nAnswer:\n{}".format(result.answer))


if __name__ == "__main__":
    main()
