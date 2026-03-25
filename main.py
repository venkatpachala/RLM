"""
main.py - RLM on your documents
================================

USAGE (3 ways):

  Way 1 - Hardcode your key below (simplest):
    Edit API_KEY = "sk-or-v1-your-key-here" in this file
    python main.py

  Way 2 - Pass everything as arguments (no prompts, no env vars needed):
    python main.py --key "sk-or-v1-..." --question "Summarize this document"

  Way 3 - Environment variable:
    On Windows CMD:    set OPENROUTER_API_KEY=sk-or-v1-...
    On Windows PS:     $env:OPENROUTER_API_KEY="sk-or-v1-..."
    On Mac/Linux:      export OPENROUTER_API_KEY=sk-or-v1-...
    Then:              python main.py

GET A FREE API KEY:
  https://openrouter.ai/keys

INSTALL DEPENDENCIES FIRST:
  pip install pymupdf requests
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
    parser.add_argument("--mock",     action="store_true",  help="Use mock LLM (no API needed)")
    parser.add_argument("--folder",   type=str, default="", help="Document folder path")
    args = parser.parse_args()

    # Resolve folder: use arg if given, else default to <script_dir>/document_pdf
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.folder:
        doc_folder = args.folder
    else:
        doc_folder = os.path.join(script_dir, "document_pdf")

    print("=" * 60)
    print("  RLM - Recursive Language Model")
    print("  github.com/lambda-calculus-LLM/lambda-RLM")
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

    # ── Step 4: Build LLM ────────────────────────────────────────────────────
    print("[Step 2/4] Setting up LLM...")

    use_mock = args.mock or not resolved_key

    if use_mock:
        if not resolved_key:
            print("  WARNING: No API key found.")
            print()
            print("  To use the real NVIDIA model, do ONE of these:")
            print("    a) Edit API_KEY = \"your-key\" at the top of main.py")
            print("    b) Run: python main.py --key \"your-key\"")
            print("    c) On Windows CMD: set OPENROUTER_API_KEY=your-key")
            print("    d) Get a free key at: https://openrouter.ai/keys")
            print()
            print("  Falling back to MockLLM (simulated responses)...")
        else:
            print("  Using MockLLM as requested (--mock flag)")
        from core.llm import MockLLM
        llm = MockLLM(context_window=CONTEXT_WINDOW_WORDS, mode="smart", verbose=True)
    else:
        print("  Model   : nvidia/llama-3.1-nemotron-70b-instruct")
        print("  Via     : OpenRouter API")
        print("  Window  : {:,} words per chunk".format(CONTEXT_WINDOW_WORDS))
        from core.llm import OpenRouterLLM
        llm = OpenRouterLLM(api_key=resolved_key, context_window=CONTEXT_WINDOW_WORDS)

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
    except Exception as e:
        print("\nCould not save result file: {}".format(e))
        print("\nAnswer:\n{}".format(result.answer))


if __name__ == "__main__":
    main()