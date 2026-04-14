# Agentic RAG System with Evaluation Framework

A production-ready Retrieval-Augmented Generation (RAG) system with **explicit query routing**, **anti-hallucination safeguards**, and a **comprehensive evaluation pipeline**.

## 🏗️ Architecture

```
                         ┌──────────────────────────────────────────────────────────────┐
                         │                    AGENTIC RAG SYSTEM                        │
                         └──────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌───────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│              │     │                   │     │   AGENTIC ROUTER    │     │                  │
│  User Query  │────>│    RETRIEVER      │────>│  (3-Layer Hybrid)   │────>│    GENERATOR     │
│              │     │                   │     │                     │     │                  │
└──────────────┘     │  • Embed query    │     │  Layer 1: Keywords  │     │  FACTUAL:        │
                     │  • FAISS search   │     │  Layer 2: Semantic  │     │    Direct answer  │
                     │  • Top-k chunks   │     │  Layer 3: TF-IDF    │     │                  │
                     │  • Score results  │     │                     │     │  SYNTHESIS:       │
                     └───────────────────┘     │  ┌───────────────┐  │     │    Multi-source   │
                              │                │  │   FACTUAL     │  │     │                  │
                              │                │  │   SYNTHESIS   │  │     │  OUT_OF_SCOPE:    │
                              │                │  │   OUT_OF_SCOPE│  │     │    Fixed refusal  │
                         Similarity            │  └───────────────┘  │     │    (no LLM call)  │
                         Scores                └─────────────────────┘     └──────────────────┘
                                                                                    │
                     ┌───────────────────┐                                          │
                     │   INGESTION       │                                   ┌──────▼──────┐
                     │                   │                                   │   Answer    │
                     │  Load → Clean     │                                   │ + Confidence│
                     │  → Chunk → Embed  │                                   │ + Sources   │
                     │  → FAISS Store    │                                   └─────────────┘
                     └───────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              EVALUATION FRAMEWORK                                            │
│                                                                                              │
│   15 Test Queries (5 Factual + 5 Synthesis + 5 Out-of-Scope)                                │
│   Metrics: Routing Accuracy | Retrieval Accuracy | ROUGE-L | Cosine Similarity               │
│   Output: Terminal Table + CSV File                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
agentic-rag-system/
│── main.py              # CLI entry point (ingest/query/evaluate/demo)
│── ingestion.py         # Document loading, chunking, embedding, FAISS storage
│── retriever.py         # FAISS-based similarity search with diversity option
│── router.py            # ★ 3-layer hybrid query router (keyword + semantic + TF-IDF)
│── generator.py         # Answer generation (Gemini API or extractive fallback)
│── evaluation.py        # Full evaluation pipeline with 15 test queries
│── utils.py             # Config, logging, text cleaning utilities
│── requirements.txt     # Python dependencies
│── README.md            # This file
│── FAILURES.md          # Failure analysis document
│── sample_docs/         # Sample documents for testing
│   ├── machine_learning_basics.txt
│   ├── deep_learning.txt
│   ├── natural_language_processing.txt
│   ├── computer_vision.txt
│   └── mlops_deployment.txt
│── vector_store.index   # FAISS index (generated after ingestion)
│── chunk_metadata.pkl   # Chunk metadata (generated after ingestion)
└── evaluation_results.csv  # Evaluation output (generated after evaluation)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ingest Documents

```bash
python main.py ingest
```

This will:
- Load all `.txt` and `.md` files from `sample_docs/`
- Clean and chunk the text (500-char chunks, 100-char overlap)
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Build and save a FAISS index

### 3. Query the System

```bash
python main.py query
```

Or run a quick demo:

```bash
python main.py demo
```

### 4. Run Evaluation

```bash
python evaluation.py
```

Or via main:

```bash
python main.py evaluate
```

## 🧠 Chunking Strategy

### Why 500-character chunks with 100-character overlap?

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Chunk Size** | 500 chars | ~3-4 sentences. Large enough to preserve paragraph-level context, small enough for precise retrieval. |
| **Overlap** | 100 chars | ~20% overlap. Ensures information at chunk boundaries is captured in both adjacent chunks. |
| **Split Strategy** | Smart boundary | Prefers splitting at sentence boundaries (`.`, `!`, `?`) to avoid mid-sentence cuts. |

**Why not sentence-level chunking?**
- Variable sentence lengths (10-500 chars) produce inconsistent similarity scores
- Single sentences often lack sufficient context for a good answer

**Why not paragraph-level chunking?**
- Paragraphs can be 50-2000+ chars — too inconsistent for retrieval
- Large chunks dilute relevant information with irrelevant content

## 🔀 Routing Logic (Detailed)

The router uses a **3-layer hybrid architecture** — NO black-box LLM routing.

### Layer 1: Keyword Rules (Deterministic)

Pattern matching against curated regex lists:

| Category | Example Patterns |
|----------|-----------------|
| **FACTUAL** | `what is`, `define`, `who invented`, `how many`, `explain` |
| **SYNTHESIS** | `compare`, `differences between`, `vs`, `pros and cons`, `evolution of` |
| **OUT_OF_SCOPE** | `capital of`, `recipe`, `world cup`, `stock price`, `weather` |

### Layer 2: Semantic Relevance (Embedding-based)

- Uses the retrieval similarity score from FAISS
- If the top retrieved chunk has similarity < 0.30, the query is likely out-of-scope
- This catches OOS queries that don't match keyword patterns

### Layer 3: TF-IDF Corpus Relevance (Statistical)

- TF-IDF model built from the document corpus vocabulary
- Query is transformed and compared against corpus
- Measures vocabulary overlap (does the query use words from our documents?)

### Decision Logic

```
1. IF oos_keywords matched AND semantic_relevance < 0.40 → OUT_OF_SCOPE
2. ELIF semantic_relevance < 0.30 → OUT_OF_SCOPE (documents don't cover this)
3. ELIF tfidf_relevance < 0.05 AND semantic_relevance < 0.35 → OUT_OF_SCOPE
4. ELIF synthesis_patterns matched → SYNTHESIS
5. ELIF multiple chunks equally relevant (broad query) → SYNTHESIS
6. ELSE → FACTUAL (default for in-scope queries)
```

**Every decision is logged with:**
- Which rules fired
- All three layer scores
- Human-readable reasoning string

### Why not use an LLM for routing?

1. **Non-deterministic**: Same query → different classifications across runs
2. **Black box**: Can't inspect why a route was chosen
3. **Slow & expensive**: API call for every query
4. **Unnecessary**: Rule-based + statistical methods handle this task well

## 📊 Evaluation Methodology

### Test Suite

| Category | Count | Purpose |
|----------|-------|---------|
| Factual | 5 | Direct answers from single chunks |
| Synthesis | 5 | Combining info from multiple sources |
| Out-of-Scope | 5 | Queries impossible to answer from docs |

### Metrics

| Metric | What it Measures | Score Range |
|--------|------------------|-------------|
| **Routing Accuracy** | Correct query classification | 0% - 100% |
| **Retrieval Accuracy** | Expected keywords in retrieved chunks | 0.0 - 1.0 |
| **ROUGE-L F1** | Longest common subsequence overlap | 0.0 - 1.0 |
| **Cosine Similarity** | Semantic similarity of answer embeddings | 0.0 - 1.0 |
| **Latency** | End-to-end query processing time | Seconds |

### Expected Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Routing Accuracy | ≥ 80% | Across all categories |
| Retrieval Accuracy | ≥ 0.70 | For in-scope queries |
| ROUGE-L F1 | ≥ 0.20 | Extractive answers match less perfectly |
| Cosine Similarity | ≥ 0.50 | Semantic alignment with expected answers |

## 🔧 Configuration

All parameters are configured in `utils.py` → `Config` dataclass:

```python
# Key parameters you can tune:
chunk_size = 500          # Characters per chunk
chunk_overlap = 100       # Overlap between chunks
top_k = 5                # Chunks to retrieve
relevance_threshold = 0.30  # Min similarity for in-scope
embedding_model = "all-MiniLM-L6-v2"  # HuggingFace model
```

### Optional: Google Gemini Integration

Set your API key to use Google Gemini for generation:

```bash
export GEMINI_API_KEY="AIzaSy..."
```

Or create a `.env` file:

```
GEMINI_API_KEY=your-gemini-key-here
LLM_MODEL=gemini-2.0-flash
```

Without an API key, the system uses **extractive generation** (selects sentences from retrieved chunks) — which is always grounded and cannot hallucinate.

## 🛡️ Anti-Hallucination Measures

1. **OUT_OF_SCOPE → Fixed response**: No LLM is called; a hardcoded refusal message is returned
2. **Strict system prompts**: Gemini prompts explicitly forbid adding information beyond the context
3. **Low temperature (0.1)**: Near-deterministic generation
4. **Extractive fallback**: When no API key is available, answers are literally extracted from documents
5. **Confidence scores**: Every answer includes a confidence indicator

## 🏃 Running the System

```bash
# Full pipeline
python main.py ingest     # Build the index
python main.py query      # Interactive Q&A
python main.py evaluate   # Run evaluation
python main.py demo       # Quick demo

# Direct evaluation
python evaluation.py
```

## 📈 Results Summary

After running `python evaluation.py`, results are saved to `evaluation_results.csv` and printed as a table. Expected results with the sample dataset:

- **Routing Accuracy**: ~87-100% (strong keyword + semantic signals)
- **Retrieval Accuracy**: ~80-100% for in-scope queries
- **ROUGE-L F1**: ~0.20-0.50 (extractive answers have partial overlap)
- **Cosine Similarity**: ~0.50-0.90 (semantic alignment is strong)

## 🔍 Failure Analysis

A comprehensive analysis of system failures, including ambiguous queries, near-boundary out-of-scope queries, and contradictory information handling, is maintained in a separate document. 

Please see the [Failure Analysis (FAILURES.md)](FAILURES.md) for detailed descriptions of common failure modes, root causes, and planned improvements.

## 🔬 Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector DB | FAISS (IndexFlatIP) | Exact search, simple, efficient for <100K vectors |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, fast, no API key needed, 384-dim |
| Routing | Hybrid (keywords + semantic + TF-IDF) | Deterministic, inspectable, no LLM dependency |
| Generation | Gemini + extractive fallback | Grounded by design, works without API key |
| Chunking | Sliding window (500/100) | Consistent size, smart sentence-boundary splitting |
| Metrics | ROUGE-L + Cosine Similarity | Content overlap + semantic alignment |

## 📝 License

MIT License — feel free to use, modify, and distribute.
