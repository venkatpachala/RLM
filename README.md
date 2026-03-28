# RLM v0

Recursive document question answering over long files using two execution styles:

- `optimized` mode: deterministic recursive chunking with LLM leaf analysis and synthesis
- `repl` mode: the LLM writes Python that decides how to recurse over the document

The project is designed around one core idea: keep the full document outside the model context window, and only expose the parts the system needs at each step...

## What This Project Does

This repository loads a long document from `document_pdf/`, turns it into an internal `Document` object, and answers a user question by recursively breaking the document into smaller chunks.

It supports:

- local inference through Ollama
- remote inference through OpenRouter
- PDF, TXT, and Markdown input files
- deterministic recursive summarization
- experimental REPL-driven recursive code generation
- optional trajectory logging for the optimized pipeline

## Default Model Setup

The current default runtime path is:

- provider: Ollama
- model: `qwen2.5:3b`

You can still use OpenRouter by passing `--key` or setting `OPENROUTER_API_KEY`.

## Project Layout

```text
rlm_v0/
├── core/
│   ├── __init__.py         # Public exports
│   ├── api.py              # Package-style programmatic API
│   ├── document.py         # Document loading, storage, slicing, chunk helpers
│   ├── llm.py              # LLM adapters: Ollama + OpenRouter
│   ├── optimized_rlm.py    # Deterministic recursive pipeline
│   ├── parser.py           # Helpers for parsing final(...) style outputs
│   ├── prompts.py          # Shared prompt builders
│   ├── repl.py             # REPL executor and sandbox namespace
│   └── rlm_system.py       # Open-ended recursive RLM orchestrator
├── document_pdf/           # Put one input document here
├── tests/                  # Test suite
├── main.py                 # CLI entry point
├── rlm_result.txt          # Output file written after each run
└── README.md
```

## Installation

Install Python dependencies:

```bash
pip install pymupdf requests llama-index-core
```

If you want to use Ollama, also install Ollama itself and pull a model:

```bash
ollama pull qwen2.5:3b
```

If you want to use OpenRouter, get an API key from:

```text
https://openrouter.ai/keys
```

## Input Documents

Supported file types:

- `.pdf`
- `.txt`
- `.md`

Put one file into the `document_pdf/` folder. The loader currently picks the first supported file in sorted order.

## How To Run

### 1. Run with the default local Ollama model

Start Ollama, make sure the model exists, then run:

```bash
ollama serve
ollama pull qwen2.5:3b
python main.py --question "Summarize the document"
```

### 2. Run with a different Ollama model

Example with `qwen2.5:7b`:

```bash
ollama pull qwen2.5:7b
python main.py --ollama-model qwen2.5:7b --question "Summarize the document"
```

### 3. Run with OpenRouter

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-..."
python main.py --question "Summarize the document"
```

Or pass the key directly:

```bash
python main.py --key "sk-or-v1-..." --question "Summarize the document"
```

### 4. Choose the execution mode

Optimized deterministic mode:

```bash
python main.py --question "Summarize the document" --mode optimized
```

REPL-driven experimental mode:

```bash
python main.py --question "Summarize the document" --mode repl
```

### 5. Save an execution trace

The optimized pipeline can write a JSONL trajectory log:

```bash
python main.py --question "Summarize the document" --log logs/run.jsonl
```

### 6. Useful runtime options

```bash
python main.py \
  --question "Find the main rules for parliamentary motions" \
  --mode optimized \
  --leaf-chunk-words 800 \
  --min-leaf-chunk-words 200 \
  --max-llm-calls 100 \
  --max-seconds 600
```

## CLI Reference

Main options:

- `--question`: question to ask about the document
- `--folder`: override the document folder path
- `--mode {optimized,repl}`: choose deterministic or REPL-driven execution
- `--ollama-model`: local Ollama model name
- `--ollama-url`: Ollama API endpoint
- `--key`: OpenRouter API key
- `--log`: JSONL trace output path for optimized mode
- `--max-llm-calls`: call budget in optimized mode
- `--max-seconds`: time budget in optimized mode
- `--leaf-chunk-words`: target leaf size in optimized mode
- `--min-leaf-chunk-words`: lower bound when adaptive backoff is needed

## End-to-End Flow

At runtime the system follows this high-level path:

```text
main.py
  -> load_document_from_folder()
  -> build LLM adapter (Ollama or OpenRouter)
  -> choose execution mode
      -> OptimizedRLMSystem
      -> or RLMSystem
  -> run question answering
  -> write result to rlm_result.txt
```

## Architecture In Depth

### 1. Entry Layer

`main.py` is the CLI boundary.

It is responsible for:

- parsing user arguments
- locating the document folder
- resolving whether to use Ollama or OpenRouter
- selecting `optimized` or `repl` mode
- printing progress information
- saving the final answer into `rlm_result.txt`

For library-style use, `core/api.py` exposes:

- `RLMConfig`
- `RLM.complete(...)`
- `RLM.acomplete(...)`

That API is the cleaner integration surface if another Python application wants to call the system directly.

### 2. Document Layer

`core/document.py` defines the document abstraction used by the rest of the system.

Responsibilities:

- load files from disk
- extract text from PDFs via PyMuPDF
- normalize whitespace
- store the full document outside the LLM prompt
- support document slicing and recursive splitting

Key methods:

- `peek(start, end)`: inspect a word range
- `split(k)`: split into `k` chunks
- `slice(start, end)`: create a sub-document from a word range
- `fits_in_window(context_window)`: check if the chunk can fit directly in model context

This layer is critical because the entire architecture depends on treating the source document as an external data structure rather than stuffing the whole text into one prompt.

### 3. LLM Adapter Layer

`core/llm.py` isolates model-provider behavior from the recursion engines.

Main classes:

- `BaseLLM`: shared accounting and interface
- `OllamaLLM`: local generation via the Ollama HTTP API
- `OpenRouterLLM`: remote generation via OpenRouter

This layer provides two important operations:

- `generate(prompt)`: used in REPL mode where the model returns Python code
- `complete_text(prompt, system_prompt=None)`: used in optimized mode where the model returns a direct textual answer

Because the execution engines talk to a common interface, the rest of the system does not need to know whether the model is local or remote.

### 4. Prompt Layer

`core/prompts.py` centralizes prompt construction.

It defines:

- `build_leaf_prompt(...)`
- `build_synthesis_prompt(...)`
- `build_repl_task_suffix(...)`

This gives the project a cleaner separation between:

- control flow and orchestration logic
- actual prompt wording

That separation is especially useful because the two pipelines ask the LLM for very different things:

- optimized mode asks for chunk analysis and answer synthesis
- REPL mode asks for executable Python control logic

### 5. Optimized Deterministic Pipeline

`core/optimized_rlm.py` is the more stable and production-oriented path.

This mode does not let the LLM control recursion. Instead, Python owns the recursion tree and the LLM is used only for local analysis and answer merging.

Core classes:

- `OptimizedRLMConfig`
- `TrajectoryLogger`
- `OptimizedRLMSystem`

Core flow:

```text
run(document, question)
  -> _process(document, depth=0)
      -> if chunk fits target leaf size:
           _summarize_leaf()
         else:
           _split_with_overlap()
           _prioritize_chunks()
           recurse over children
           _synthesize_answers()
  -> return RLMResult
```

Important design features:

- deterministic chunking
- overlap-aware splitting so information is less likely to fall across hard chunk boundaries
- lexical chunk prioritization for lookup-style questions
- focused excerpt extraction for fact-finding questions
- call budgets and wall-clock budgets
- optional final synthesis
- JSONL trace logging

This mode is easier to reason about operationally because:

- recursion depth is capped
- LLM call count is budgeted
- time is budgeted
- chunk sizes are controlled by config

### 6. REPL-Driven Recursive Pipeline

`core/rlm_system.py` and `core/repl.py` implement the more open-ended RLM variant.

In this mode, the LLM is not just answering questions. It is writing Python code that decides how to decompose the problem.

High-level flow:

```text
RLMSystem.run()
  -> _call(document, question, depth=0)
      -> create REPLNamespace
      -> REPLExecutor.run(...)
          -> build prompt
          -> ask LLM for Python code
          -> exec(code, sandboxed namespace)
          -> generated code may call:
               split()
               peek()
               sub_call()
               merge()
               final()
```

Key parts:

- `REPLNamespace`: the restricted execution environment
- `REPLExecutor`: the turn loop that prompts, runs code, and checks for completion
- `RLMSystem`: the recursive orchestrator that injects `sub_call`

This architecture is more flexible but also less predictable.

Failure modes include:

- the model writes invalid Python
- the model recurses poorly
- the model forgets to call `final(...)`
- the model chooses inefficient decompositions

That tradeoff is the reason the repository includes both modes rather than only one.

### 7. Result Model

`core/rlm_system.py` defines `RLMResult`, the shared result container returned by the execution systems.

It includes:

- `answer`
- `llm_calls`
- `elapsed_sec`
- `max_depth`
- `succeeded`
- `failure`

This common result shape makes it easier for `main.py` and `core/api.py` to treat both execution styles consistently.

## Detailed Request Lifecycle

Here is the lifecycle of a typical optimized run:

```text
1. User runs main.py with a question
2. main.py loads the first supported file from document_pdf/
3. The file is converted to a Document object
4. main.py builds an LLM adapter
5. OptimizedRLMSystem.run() begins
6. The root document is recursively broken into smaller chunks
7. Leaf chunks are sent to the model with a direct-answer prompt
8. Leaf answers are merged upward using synthesis prompts
9. A final RLMResult is returned
10. main.py writes the answer to rlm_result.txt
```

And for REPL mode:

```text
1. User runs main.py with --mode repl
2. main.py loads the document and builds an LLM adapter
3. RLMSystem.run() starts the root recursive node
4. REPLExecutor prompts the model with document metadata and history
5. The model returns Python code
6. The code executes in a controlled namespace
7. The code may split the document and recursively call sub_call()
8. The code eventually calls final(answer)
9. The answer bubbles back up to the root result
10. main.py writes the answer to rlm_result.txt
```

## Programmatic Usage

You can also call the system directly from Python:

```python
from core.api import RLM, RLMConfig
from core.llm import OllamaLLM

llm = OllamaLLM(model="qwen2.5:3b", context_window=6000)
config = RLMConfig(max_depth=6, leaf_chunk_words=800)
rlm = RLM(llm=llm, config=config, verbose=True)

result = rlm.complete(
    context="Long text goes here...",
    question="Summarize the main points",
    document_name="example.txt",
)

print(result.answer)
```

## Output Files

After each CLI run, the final answer is written to:

```text
rlm_result.txt
```

If `--log` is enabled in optimized mode, a JSONL trajectory file is also written.

## Known Notes

- The loader currently chooses the first supported file in `document_pdf/`.
- Large documents can result in multiple recursive leaf calls even in optimized mode.
- Ollama must be running locally if you use the default local path.
- OpenRouter is only used when a key is explicitly provided.
- The REPL mode is more experimental than the optimized mode.

## Recommended Starting Point

If you are exploring the codebase for the first time, start here:

1. `main.py`
2. `core/document.py`
3. `core/llm.py`
4. `core/optimized_rlm.py`
5. `core/repl.py`
6. `core/rlm_system.py`

That sequence follows the same order the system uses at runtime.
