## RLM v0

Recursive document question answering for large PDF and text files.

### What this project does

This project keeps the full document outside the LLM context window, then:

- loads the source file into a `Document`
- recursively splits large inputs into manageable chunks
- asks the LLM to answer at the leaf level
- synthesizes chunk answers into one final result

It supports two execution styles:

- `optimized`: deterministic recursion with targeted retrieval and synthesis
- `repl`: LLM-generated recursive control flow inside a restricted Python sandbox

### Key improvements in this version

- reusable package API via `core.api.RLM`
- async entrypoint via `RLM.acomplete(...)`
- configurable primary and recursive models
- generic OpenAI-compatible backend support
- stronger optimized retrieval with chunk prioritization
- richer safe REPL helpers like regex and JSON parsing
- trajectory logging for recursive runs

### CLI examples

Use mock mode:

```bash
python main.py --mock --question "Summarize the main points"
```

Use OpenRouter:

```bash
python main.py --key "sk-or-v1-..." --model "nvidia/llama-3.1-nemotron-70b-instruct" --recursive-model "nvidia/llama-3.1-nemotron-70b-instruct" --mode optimized --question "Summarize the main points"
```

Use any OpenAI-compatible endpoint:

```bash
python main.py --backend openai-compatible --api-url "https://your-endpoint/v1/chat/completions" --key "..." --model "gpt-4o-mini" --recursive-model "gpt-4o-mini" --question "What is the key finding?"
```

### Python API

```python
from core.api import RLM, RLMConfig
from core.llm import MockLLM

llm = MockLLM()
rlm = RLM(llm=llm, config=RLMConfig(leaf_chunk_words=800))
result = rlm.complete("long context goes here", "Summarize the key points")
print(result.answer)
```

### Architecture

- `core/document.py`: document model, slicing, splitting, PDF/TXT loading
- `core/llm.py`: model backends and completion abstraction
- `core/optimized_rlm.py`: deterministic recursive engine
- `core/repl.py`: restricted REPL for agentic recursive control
- `core/rlm_system.py`: orchestrator for REPL mode
- `core/api.py`: reusable package-style entrypoint
- `core/prompts.py`: shared prompt builders
- `core/parser.py`: final-answer parsing helpers

### Tests

```bash
python -m pytest tests/test_rlm.py
```
