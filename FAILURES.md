# Failure Analysis — Agentic RAG System

This document identifies and analyzes failure cases in the Agentic RAG system,
explaining **what failed**, **why it failed**, and **how to improve** each case.

---

## Failure Case 1: Ambiguous Queries at the Factual/Synthesis Boundary

### What Failed

Queries like **"What are the types of machine learning?"** are ambiguous — they could be:
- **FACTUAL**: The user wants a simple list (supervised, unsupervised, reinforcement)
- **SYNTHESIS**: The user wants a detailed comparison of each type

The router sometimes classifies these as FACTUAL when a SYNTHESIS-style answer would be more helpful, resulting in an overly brief response that misses important context.

### Why It Failed

The keyword pattern matching in Layer 1 of the router triggers on `"what are"` (a FACTUAL pattern) before considering that the breadth of the question warrants synthesis. The router prioritizes the first matching pattern category rather than analyzing the **scope** of what's being asked.

Additionally, the retrieval layer returns many highly-relevant chunks from the same document, which looks like a FACTUAL signal (one source is sufficient) rather than a SYNTHESIS signal.

### How to Improve

1. **Query Complexity Analysis**: Add a sub-layer that estimates query complexity by:
   - Counting the number of distinct concepts in the query
   - Checking if the answer would require multiple paragraphs
   - Using a simple classifier trained on (query, best_response_type) pairs

2. **Response Length Heuristic**: If the top-k retrieved chunks cover significantly different sub-topics (measured by inter-chunk cosine distance), promote to SYNTHESIS even if keyword rules say FACTUAL.

3. **User Feedback Loop**: Allow the user to say "give me more detail" to dynamically upgrade a FACTUAL response to SYNTHESIS.

---

## Failure Case 2: Near-Boundary Out-of-Scope Queries

### What Failed

Queries that are **tangentially related** to the corpus but not directly answerable cause problems. For example:

- **"What programming language is best for machine learning?"** — Our documents discuss ML concepts but don't compare programming languages.
- **"How much does it cost to train GPT-4?"** — Documents mention GPT-4's parameter count but not training costs.

These queries achieve moderate semantic relevance scores (0.25-0.35) that hover near the OUT_OF_SCOPE threshold (0.30), leading to inconsistent classification.

### Why It Failed

The relevance threshold is a hard boundary (0.30). Queries with scores in the 0.25-0.35 range are in a "gray zone" where small changes in query phrasing can flip the classification. This happens because:

1. **Embedding similarity is continuous** — there's no natural boundary between "relevant" and "irrelevant" in embedding space.
2. **Partial topic overlap** — the query shares vocabulary with the corpus (e.g., "machine learning", "GPT-4") which inflates the similarity score, even though the specific question can't be answered.
3. **TF-IDF layer agrees** — because the query uses corpus vocabulary, the TF-IDF layer also gives a moderate score, reinforcing the false positive.

### How to Improve

1. **Soft Boundaries with Calibration**: Replace the hard threshold with a calibrated probability model:
   - Train a logistic regression on pairs of (similarity_score, is_answerable)
   - Output a probability rather than a binary decision
   - Use the probability to modulate answer confidence

2. **Answer Verification Step**: After generating an answer, add a verification step that checks:
   - Does the answer directly address the question?
   - Are there specific facts/numbers in the answer that match the query?
   - If verification fails, downgrade to OUT_OF_SCOPE

3. **Negative Example Training**: Curate a set of near-boundary OOS examples during development and use them to fine-tune the threshold and add specific router rules.

4. **Abstention Mechanism**: Instead of binary in-scope/out-of-scope, add an "UNCERTAIN" category with a response like: *"I found related information but may not be able to fully answer this question. Here's what I found: ..."*

---

## Failure Case 3: Contradictory Information Across Documents

### What Failed

When documents contain **contradictory or conflicting information**, the system may produce misleading answers without flagging the conflict. For example:

- Document A states: *"ResNet achieved 3.57% top-5 error on ImageNet."*
- If another document were to state a different error rate for ResNet, the system would pick whichever chunk scores highest without noting the discrepancy.

Currently, the system retrieves chunks independently and passes them to the generator without cross-referencing. The extractive generator simply selects the highest-scoring sentences, which may come from only one source.

### Why It Failed

1. **No contradiction detection**: The retrieval and generation pipeline treats each chunk as an independent piece of evidence. There's no mechanism to compare facts across chunks.

2. **Winner-takes-all retrieval**: The highest-scoring chunk dominates the answer. If two chunks present different facts about the same topic, only one is used.

3. **Extractive generation limitation**: The extractive generator (fallback mode) cannot synthesize or identify conflicts — it simply copies sentences.

### How to Improve

1. **Contradiction Detection Module**: Add a post-retrieval step that:
   - Groups retrieved chunks by topic (using clustering or keyword matching)
   - For each topic group, extracts key claims (entity + attribute + value)
   - Compares claims across sources to detect conflicts
   - Flags contradictions in the answer

2. **Multi-Source Attribution**: Instead of a single answer, present information organized by source:
   ```
   According to [Document A]: ResNet achieved 3.57% error.
   According to [Document B]: ResNet achieved 3.6% error.
   Note: These sources show slightly different figures.
   ```

3. **Confidence-Weighted Voting**: When multiple chunks provide different answers to the same factual question, use a voting mechanism weighted by:
   - Chunk relevance score
   - Source recency (if timestamps available)
   - Source authority (if metadata available)

4. **Provenance Tracking**: Attach full provenance to every fact in the answer, allowing the user to verify claims independently.

---

## Additional Observations

### Edge Case: Multi-Part Queries

Queries like **"What is BERT and how does it compare to GPT?"** contain both a FACTUAL sub-query ("What is BERT?") and a SYNTHESIS sub-query ("compare to GPT"). The current system treats the entire query as one unit, which forces a single classification. 

**Improvement**: Implement query decomposition that splits multi-part queries into sub-queries, routes each independently, and combines the answers.

### Edge Case: Numerical/Statistical Queries

Questions like **"How many parameters does GPT-3 have?"** require precise numerical extraction. The extractive generator may include the correct number in a longer sentence but the ROUGE score may be low because of surrounding text.

**Improvement**: Add a number extraction post-processing step for queries containing "how many", "how much", or "what number" patterns.

### Edge Case: Temporal Queries

Questions about **when** something happened or the **latest** version may retrieve outdated chunks when documents have different publication dates.

**Improvement**: Add timestamp metadata to chunks and implement recency-aware retrieval for temporal queries.

---

## Summary Table

| Failure Case | Root Cause | Severity | Fix Complexity |
|---|---|---|---|
| Ambiguous FACTUAL/SYNTHESIS | Rigid keyword precedence | Medium | Medium |
| Near-boundary OOS | Hard threshold on continuous scores | High | Medium |
| Contradictory information | No cross-chunk verification | High | High |
| Multi-part queries | No query decomposition | Medium | Medium |
| Numerical extraction | Extractive generation verbosity | Low | Low |
| Temporal queries | No timestamp-aware retrieval | Low | Medium |

---

*This failure analysis is based on systematic testing with the evaluation framework. The identified patterns provide a clear roadmap for system improvement.*
