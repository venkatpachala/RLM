# RLM Implementation — Standard Recursive Language Model

## What this is

This is a complete implementation of **standard RLM** (Recursive Language Model)
from Zhang et al. 2026. It is **NOT λ-RLM** (that is the follow-up system
with formal guarantees — we implement that next).

## Quick start

```bash
# 1. Install dependencies
pip install pymupdf requests

# 2. Get a free OpenRouter API key
#    https://openrouter.ai/keys

# 3. Set your key
export OPENROUTER_API_KEY="sk-or-..."

# 4. Drop your document in the folder
cp your_document.pdf document_pdf/

# 5. Run
python main.py
```

## Project structure

```
rlm_real/
├── document_pdf/        ← DROP YOUR DOCUMENT HERE
├── main.py              ← Single entry point — run this
├── core/
│   ├── document.py      ← Document class + PDF loader
│   ├── llm.py           ← OpenRouter (NVIDIA) + MockLLM
│   ├── repl.py          ← The open-ended REPL loop
│   └── rlm_system.py    ← Orchestrator + recursion wiring
└── rlm_result.txt       ← Output saved here after each run
```

## How the system flows (complete trace)

```
You run main.py
│
├─ [1] load_document_from_folder("document_pdf/")
│       Reads PDF → extracts text → creates Document object
│       Full text stored in memory, NOT given to LLM yet
│
├─ [2] OpenRouterLLM(api_key, context_window=6000)
│       Wraps NVIDIA Llama-3.1 Nemotron 70B via OpenRouter API
│
├─ [3] You type your question
│
└─ [4] RLMSystem.run(document, question)
         │
         └─ _call(document, depth=0)
              │
              ├─ Creates REPLNamespace
              │    P = document (120K words, stored here)
              │    sub_call = lambda doc, q: _call(doc, q, depth+1)
              │    split, peek, merge, final also available
              │
              └─ REPLExecutor.run()
                   │
                   ├─ [Turn 1] Build prompt:
                   │    "Document: 120K words, Preview: first 150 words..."
                   │    "Question: ..."
                   │    (document NOT fully in prompt — only metadata)
                   │
                   ├─ [Turn 1] Call LLM → gets back Python code:
                   │    chunks = split(P, 4)
                   │    results = [sub_call(c, question) for c in chunks]
                   │    final(merge(results))
                   │
                   ├─ [Turn 1] exec(code, namespace)
                   │    split(P, 4) → [30K, 30K, 30K, 30K chunks]
                   │    sub_call(chunk1, q) → _call(chunk1, depth=1)
                   │                            └─ REPL loop at depth 1
                   │                               LLM sees full 30K text
                   │                               LLM calls final("answer1")
                   │    sub_call(chunk2, q) → _call(chunk2, depth=1) ...
                   │    sub_call(chunk3, q) → ...
                   │    sub_call(chunk4, q) → ...
                   │    merge(["answer1","answer2","answer3","answer4"])
                   │    final(merged_answer)  ← loop exits here
                   │
                   └─ Returns final merged answer
```

## RLM vs λ-RLM comparison

| Property              | Standard RLM (this code) | λ-RLM (next implementation) |
|-----------------------|--------------------------|------------------------------|
| Control flow          | LLM generates Python     | Pre-verified combinators     |
| Termination           | Not guaranteed           | Proven (Theorem 1)           |
| Cost before run       | Unknown                  | Exact (N = k*^d + 1)         |
| Failure modes         | 4 (see below)            | 0 (all eliminated)           |
| Code crashes          | Possible                 | Impossible (no generated code)|
| 8B model accuracy     | ~14% avg                 | ~36% avg (+21.9 points)      |
| Latency               | High (5-12 turns)        | 4x faster (1 combinator run) |

## The 4 RLM failure modes (demonstrated in this code)

1. **Non-termination**: LLM calls `sub_call(P, q)` with same-size doc → infinite loop
2. **Code errors**: LLM writes `chuncks` (typo) → NameError → exec() fails
3. **Unpredictable cost**: LLM decides k=2 or k=50 unpredictably → bill varies 10x
4. **Coding tax**: 7B model spends capacity writing Python, not reasoning

## LLM used

**NVIDIA Llama-3.1 Nemotron 70B Instruct** via OpenRouter

- Free tier available at openrouter.ai
- 131K token context window
- Strong instruction following + code generation
- Good at producing clean Python for the REPL

## Supported document types

- `.pdf` — extracted via PyMuPDF
- `.txt` — plain text files
- `.md` — markdown files

Drop one file in `document_pdf/` and run `main.py`.