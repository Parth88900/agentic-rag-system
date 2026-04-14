"""
evaluation.py - Evaluation Framework for the Agentic RAG System.

This script runs a comprehensive evaluation pipeline:
  1. Defines 15 test queries (5 factual, 5 synthesis, 5 out-of-scope)
  2. Runs each query through the full RAG pipeline
  3. Measures multiple metrics:
     - Routing Accuracy: Did the router classify correctly?
     - Retrieval Accuracy: Were the right chunks retrieved?
     - Answer Quality: ROUGE-L and cosine similarity scores
  4. Outputs results as a formatted table and CSV file

EVALUATION METHODOLOGY:
═══════════════════════
For each test query, we define:
  - expected_type: The correct routing classification
  - expected_keywords: Keywords that should appear in the answer (for retrieval validation)
  - expected_answer: Reference answer for quality metrics
  - expected_sources: Documents that should be retrieved

Metrics Explained:
  - ROUGE-L: Measures longest common subsequence between generated and reference answers.
    Score range: 0-1, higher is better. Captures content overlap regardless of word order.
  - Cosine Similarity: Measures semantic similarity between generated and reference answer
    embeddings. Score range: -1 to 1 (typically 0-1), higher means more semantically similar.
  - Routing Accuracy: Binary — did the router pick the correct query type?
  - Retrieval Accuracy: Checks if expected keywords appear in retrieved chunks.

Run: python evaluation.py
"""

import sys
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from rouge_score import rouge_scorer

from utils import config, logger, console
from ingestion import IngestionPipeline, EmbeddingGenerator, VectorStoreManager
from retriever import Retriever
from router import QueryRouter, QueryType
from generator import AnswerGenerator, OUT_OF_SCOPE_RESPONSE


# ============================================================
# Test Query Definitions
# ============================================================
@dataclass
class TestQuery:
    """
    A test query with expected outputs for evaluation.
    
    Attributes:
        query: The test question.
        expected_type: Expected routing classification.
        expected_answer: Reference answer for quality metrics.
        expected_keywords: Keywords that should appear in retrieved chunks
                          (for retrieval accuracy).
        expected_sources: Documents that should be in the retrieval results.
        category: Human-readable category label.
    """
    query: str
    expected_type: QueryType
    expected_answer: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    category: str = ""


# ── 5 FACTUAL Queries ──
# These have direct answers in specific documents.
FACTUAL_QUERIES = [
    TestQuery(
        query="What is the maximum fine for violating prohibited AI practices under the EU AI Act?",
        expected_type=QueryType.FACTUAL,
        expected_answer=(
            "The maximum fine for violating prohibited AI practices is up to €35 million or 7% of a company’s worldwide annual turnover."
        ),
        expected_keywords=["€35 million", "7%", "worldwide annual turnover", "maximum fine"],
        expected_sources=["Document_1_Policy_Report.txt", "Document_2_News_Article.txt"],
        category="FACTUAL"
    ),
    TestQuery(
        query="Are social scoring systems allowed under the EU AI Act?",
        expected_type=QueryType.FACTUAL,
        expected_answer=(
            "No, social scoring systems by governments are strictly prohibited under the Unacceptable Risk classification."
        ),
        expected_keywords=["social scoring", "prohibited", "unacceptable risk"],
        expected_sources=["Document_1_Policy_Report.txt"],
        category="FACTUAL"
    ),
    TestQuery(
        query="Who authored the memo regarding open-source exemptions?",
        expected_type=QueryType.FACTUAL,
        expected_answer=(
            "The memo regarding open-source exemptions was authored by CRAID (Coalition for Responsible Artificial Intelligence Development)."
        ),
        expected_keywords=["CRAID", "Coalition for Responsible Artificial Intelligence Development"],
        expected_sources=["Document_3_Stakeholder_Memo.txt"],
        category="FACTUAL"
    ),
    TestQuery(
        query="What do the transparency requirements dictate for generative models?",
        expected_type=QueryType.FACTUAL,
        expected_answer=(
            "Transparency requirements dictate that users must always know when they are interacting with AI, particularly for generative models, which may involve watermarking outputs."
        ),
        expected_keywords=["interacting with AI", "watermarking", "transparency requirements"],
        expected_sources=["Document_4_Technical_Brief.txt"],
        category="FACTUAL"
    ),
    TestQuery(
        query="What are the transparency requirements for Limited Risk AI systems?",
        expected_type=QueryType.FACTUAL,
        expected_answer=(
            "Limited Risk systems, such as chatbots and deepfakes, are subject to transparency obligations where users must be informed they are interacting with an AI."
        ),
        expected_keywords=["transparency obligations", "informed"],
        expected_sources=["Document_1_Policy_Report.txt"],
        category="FACTUAL"
    ),
]

# ── 5 SYNTHESIS Queries ──
# These require combining information from multiple chunks or documents.
SYNTHESIS_QUERIES = [
    TestQuery(
        query="Compare the penalty amounts mentioned in the Policy Report and the News Article.",
        expected_type=QueryType.SYNTHESIS,
        expected_answer=(
            "The Policy Report states that the maximum fine is up to €35 million or 7% of a company’s worldwide annual turnover. In contrast, the News Article states that the maximum fines are capped at €30 million or 6% of global annual revenue. This is a clear contradiction."
        ),
        expected_keywords=["€35 million", "7%", "€30 million", "6%", "contradict"],
        expected_sources=["Document_1_Policy_Report.txt", "Document_2_News_Article.txt"],
        category="SYNTHESIS"
    ),
    TestQuery(
        query="Do the documents agree on whether real-time biometric categorization is completely prohibited?",
        expected_type=QueryType.SYNTHESIS,
        expected_answer=(
            "No, the documents contradict each other. The Policy Report states there are absolutely no exceptions for real-time biometric categorization. However, the News Article states that it can be utilized by law enforcement under strict judicial authorization for targeted cases like kidnappings or terrorism."
        ),
        expected_keywords=["contradict", "no exceptions", "law enforcement", "judicial authorization", "kidnappings"],
        expected_sources=["Document_1_Policy_Report.txt", "Document_2_News_Article.txt"],
        category="SYNTHESIS"
    ),
    TestQuery(
        query="Summarize the different viewpoints on open-source model exemptions presented in the Stakeholder Memo and the Technical Brief.",
        expected_type=QueryType.SYNTHESIS,
        expected_answer=(
            "The Stakeholder Memo (CRAID) strongly argues that open-source AI models should be completely exempt from the AI Act’s requirements regardless of size or compute thresholds. Conversely, the Technical Brief states that open-source models are exempt UNLESS they pose a systemic risk or are put into service commercially."
        ),
        expected_keywords=["completely exempt", "systemic risk", "service commercially", "Stakeholder Memo", "Technical Brief"],
        expected_sources=["Document_3_Stakeholder_Memo.txt", "Document_4_Technical_Brief.txt"],
        category="SYNTHESIS"
    ),
    TestQuery(
        query="What are the obligations for high-risk AI systems mentioned across the documents?",
        expected_type=QueryType.SYNTHESIS,
        expected_answer=(
            "High-risk AI systems must undergo strict conformity assessments before market entry. Additionally, they must implement strict data governance (unbiased datasets), automatic event logging for transparency, and human oversight mechanisms, including a 'kill switch' to override or stop AI operations."
        ),
        expected_keywords=["conformity assessments", "data governance", "logging", "human oversight", "kill switch"],
        expected_sources=["Document_1_Policy_Report.txt", "Document_4_Technical_Brief.txt"],
        category="SYNTHESIS"
    ),
    TestQuery(
        query="If a developer creates an open-source model and a separate company integrates it into a commercial product, who is responsible for compliance according to the documents?",
        expected_type=QueryType.SYNTHESIS,
        expected_answer=(
            "According to the Stakeholder Memo, open-source developers believe it should be the integrating end-user's responsibility. The Technical Brief officially confirms this, stating that the commercializing entity becomes the 'provider' and must assume full compliance obligations for the entire system, regardless of its open-source origins."
        ),
        expected_keywords=["integrating entity", "provider", "full compliance obligations", "end-user's responsibility"],
        expected_sources=["Document_3_Stakeholder_Memo.txt", "Document_4_Technical_Brief.txt"],
        category="SYNTHESIS"
    ),
]

# ── 5 OUT_OF_SCOPE Queries ──
# These cannot be answered from the AI/ML document corpus.
OUT_OF_SCOPE_QUERIES = [
    TestQuery(
        query="What is the capital of France?",
        expected_type=QueryType.OUT_OF_SCOPE,
        expected_answer=OUT_OF_SCOPE_RESPONSE,
        expected_keywords=[],
        expected_sources=[],
        category="OUT_OF_SCOPE"
    ),
    TestQuery(
        query="How do I cook pasta carbonara?",
        expected_type=QueryType.OUT_OF_SCOPE,
        expected_answer=OUT_OF_SCOPE_RESPONSE,
        expected_keywords=[],
        expected_sources=[],
        category="OUT_OF_SCOPE"
    ),
    TestQuery(
        query="Who won the FIFA World Cup in 2022?",
        expected_type=QueryType.OUT_OF_SCOPE,
        expected_answer=OUT_OF_SCOPE_RESPONSE,
        expected_keywords=[],
        expected_sources=[],
        category="OUT_OF_SCOPE"
    ),
    TestQuery(
        query="What are the symptoms of diabetes?",
        expected_type=QueryType.OUT_OF_SCOPE,
        expected_answer=OUT_OF_SCOPE_RESPONSE,
        expected_keywords=[],
        expected_sources=[],
        category="OUT_OF_SCOPE"
    ),
    TestQuery(
        query="Tell me about the stock price of Tesla today.",
        expected_type=QueryType.OUT_OF_SCOPE,
        expected_answer=OUT_OF_SCOPE_RESPONSE,
        expected_keywords=[],
        expected_sources=[],
        category="OUT_OF_SCOPE"
    ),
]

# All test queries combined
ALL_TEST_QUERIES = FACTUAL_QUERIES + SYNTHESIS_QUERIES + OUT_OF_SCOPE_QUERIES


# ============================================================
# Evaluation Metrics
# ============================================================
class EvaluationMetrics:
    """
    Computes evaluation metrics for the RAG system.
    
    Metrics:
      - ROUGE-L: Longest common subsequence overlap (content coverage)
      - Cosine Similarity: Semantic similarity via embeddings
      - Routing Accuracy: Correct query classification
      - Retrieval Accuracy: Correct chunks retrieved
    """
    
    def __init__(self):
        """Initialize metric calculators."""
        # ROUGE scorer — we use ROUGE-L (longest common subsequence)
        # ROUGE-L captures content overlap regardless of exact word order
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Embedding model for cosine similarity
        self.embedder = None  # Lazy initialization
    
    def _get_embedder(self):
        """Lazy-load the embedding model for cosine similarity."""
        if self.embedder is None:
            self.embedder = EmbeddingGenerator()
        return self.embedder
    
    def compute_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores between generated and reference answers.
        
        Returns dict with rouge1, rouge2, and rougeL F1 scores.
        """
        scores = self.rouge.score(reference, generated)
        return {
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_f1": scores["rougeL"].fmeasure,
        }
    
    def compute_cosine_similarity(self, generated: str, reference: str) -> float:
        """
        Compute cosine similarity between generated and reference answer embeddings.
        
        Uses sentence-transformers to encode both texts into dense vectors,
        then computes cosine similarity between them.
        
        Returns:
            Cosine similarity score (typically 0-1 for related texts).
        """
        embedder = self._get_embedder()
        gen_emb = embedder.embed_query(generated)
        ref_emb = embedder.embed_query(reference)
        similarity = sklearn_cosine(gen_emb, ref_emb)
        return float(similarity[0][0])
    
    def compute_routing_accuracy(
        self,
        predicted_type: QueryType,
        expected_type: QueryType
    ) -> bool:
        """Check if routing classification is correct."""
        return predicted_type == expected_type
    
    def compute_retrieval_accuracy(
        self,
        retrieved_texts: List[str],
        expected_keywords: List[str]
    ) -> float:
        """
        Compute retrieval accuracy based on keyword presence in retrieved chunks.
        
        For each expected keyword, check if it appears (case-insensitive) in
        any of the retrieved chunk texts. The score is the fraction of expected
        keywords found.
        
        Args:
            retrieved_texts: List of retrieved chunk text strings.
            expected_keywords: Keywords that should be present.
        
        Returns:
            Fraction of expected keywords found (0.0 to 1.0).
        """
        if not expected_keywords:
            return 1.0  # No keywords expected (e.g., OOS queries)
        
        combined_text = " ".join(retrieved_texts).lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
        return found / len(expected_keywords)


# ============================================================
# Evaluation Pipeline
# ============================================================
class EvaluationPipeline:
    """
    Runs the full evaluation pipeline on all test queries.
    
    Process:
    1. Ensure documents are ingested (run ingestion if needed)
    2. Initialize retriever, router, and generator
    3. For each test query:
       a. Retrieve relevant chunks
       b. Route the query
       c. Generate an answer
       d. Compute all metrics
    4. Output results as table and CSV
    """
    
    def __init__(self):
        """Initialize evaluation components."""
        self.metrics = EvaluationMetrics()
        self.results = []
    
    def _ensure_ingestion(self) -> Tuple[Retriever, QueryRouter, AnswerGenerator]:
        """
        Ensure documents are ingested and return initialized components.
        
        If the FAISS index doesn't exist on disk, runs the ingestion pipeline.
        """
        import os
        
        # Check if index exists
        if not os.path.exists(config.faiss_index_path) or \
           not os.path.exists(config.metadata_path):
            console.print("[yellow]Index not found. Running ingestion pipeline...[/yellow]")
            pipeline = IngestionPipeline()
            vector_store = pipeline.run(save=True)
            embedder = pipeline.embedder
        else:
            console.print("[green]Loading existing index...[/green]")
            embedder = EmbeddingGenerator()
            vector_store = VectorStoreManager(dim=embedder.dim)
            vector_store.load()
        
        # Initialize components
        retriever = Retriever(embedder=embedder, vector_store=vector_store)
        router = QueryRouter(corpus_texts=retriever.get_all_chunk_texts())
        generator = AnswerGenerator()
        
        return retriever, router, generator
    
    def run(self, test_queries: List[TestQuery] = None) -> pd.DataFrame:
        """
        Run evaluation on all test queries.
        
        Args:
            test_queries: List of TestQuery objects. Defaults to ALL_TEST_QUERIES.
        
        Returns:
            pandas DataFrame with evaluation results.
        """
        test_queries = test_queries or ALL_TEST_QUERIES
        
        console.rule("[bold blue]RAG System Evaluation[/bold blue]")
        console.print(f"Total test queries: {len(test_queries)}")
        console.print(f"  Factual: {sum(1 for q in test_queries if q.expected_type == QueryType.FACTUAL)}")
        console.print(f"  Synthesis: {sum(1 for q in test_queries if q.expected_type == QueryType.SYNTHESIS)}")
        console.print(f"  Out-of-Scope: {sum(1 for q in test_queries if q.expected_type == QueryType.OUT_OF_SCOPE)}")
        console.print()
        
        # Initialize components
        retriever, router, generator = self._ensure_ingestion()
        
        # Set the embedder for cosine similarity (reuse the same model)
        self.metrics.embedder = retriever.embedder
        
        results = []
        
        for i, tq in enumerate(test_queries):
            console.print(f"[{i+1}/{len(test_queries)}] Evaluating: [bold]{tq.query}[/bold]")
            
            start_time = time.time()
            
            # ── Step 1: Retrieve ──
            if tq.expected_type == QueryType.SYNTHESIS:
                retrieval_result = retriever.retrieve_with_diversity(tq.query)
            else:
                retrieval_result = retriever.retrieve(tq.query)
            
            # ── Step 2: Route ──
            routing_result = router.route(
                tq.query,
                retrieval_top_score=retrieval_result.top_score,
                retrieval_avg_score=retrieval_result.avg_score
            )
            
            # ── Step 3: Generate ──
            answer = generator.generate(tq.query, routing_result, retrieval_result)
            
            elapsed = time.time() - start_time
            
            # ── Step 4: Compute Metrics ──
            # Routing accuracy
            routing_correct = self.metrics.compute_routing_accuracy(
                routing_result.query_type, tq.expected_type
            )
            
            # Retrieval accuracy
            retrieved_texts = [chunk.text for chunk, _ in retrieval_result.chunks]
            retrieval_accuracy = self.metrics.compute_retrieval_accuracy(
                retrieved_texts, tq.expected_keywords
            )
            
            # Answer quality metrics
            rouge_scores = self.metrics.compute_rouge(answer.answer, tq.expected_answer)
            cosine_sim = self.metrics.compute_cosine_similarity(answer.answer, tq.expected_answer)
            
            # ── Step 5: Record Results ──
            result = {
                "query": tq.query,
                "category": tq.category,
                "expected_type": tq.expected_type.value,
                "predicted_type": routing_result.query_type.value,
                "routing_correct": routing_correct,
                "routing_confidence": routing_result.confidence,
                "retrieval_accuracy": retrieval_accuracy,
                "retrieval_top_score": retrieval_result.top_score,
                "rouge1_f1": rouge_scores["rouge1_f1"],
                "rouge2_f1": rouge_scores["rouge2_f1"],
                "rougeL_f1": rouge_scores["rougeL_f1"],
                "cosine_similarity": cosine_sim,
                "answer_confidence": answer.confidence,
                "generation_method": answer.generation_method,
                "sources_used": ", ".join(answer.sources) if answer.sources else "N/A",
                "latency_sec": elapsed,
                "generated_answer": answer.answer[:200],  # Truncate for readability
            }
            results.append(result)
            
            # Log progress
            status = "✅" if routing_correct else "❌"
            console.print(
                f"  {status} Route: {routing_result.query_type.value} "
                f"(expected: {tq.expected_type.value}) | "
                f"ROUGE-L: {rouge_scores['rougeL_f1']:.3f} | "
                f"Cosine: {cosine_sim:.3f} | "
                f"Retrieval: {retrieval_accuracy:.2f} | "
                f"Time: {elapsed:.2f}s"
            )
        
        # ── Build DataFrame ──
        df = pd.DataFrame(results)
        self.results = results
        
        return df
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Print a comprehensive summary of evaluation results.
        
        Shows:
        - Overall metrics
        - Per-category breakdown
        - Best and worst performing queries
        """
        console.rule("[bold blue]Evaluation Summary[/bold blue]")
        
        # ── Overall Metrics ──
        console.print("\n[bold]Overall Metrics:[/bold]")
        console.print(f"  Routing Accuracy:    {df['routing_correct'].mean():.1%}")
        console.print(f"  Retrieval Accuracy:  {df['retrieval_accuracy'].mean():.3f}")
        console.print(f"  Avg ROUGE-1 F1:      {df['rouge1_f1'].mean():.3f}")
        console.print(f"  Avg ROUGE-2 F1:      {df['rouge2_f1'].mean():.3f}")
        console.print(f"  Avg ROUGE-L F1:      {df['rougeL_f1'].mean():.3f}")
        console.print(f"  Avg Cosine Sim:      {df['cosine_similarity'].mean():.3f}")
        console.print(f"  Avg Latency:         {df['latency_sec'].mean():.2f}s")
        
        # ── Per-Category Breakdown ──
        console.print("\n[bold]Per-Category Breakdown:[/bold]")
        for category in ["FACTUAL", "SYNTHESIS", "OUT_OF_SCOPE"]:
            cat_df = df[df["category"] == category]
            if len(cat_df) > 0:
                console.print(f"\n  [{category}] ({len(cat_df)} queries)")
                console.print(f"    Routing Accuracy:   {cat_df['routing_correct'].mean():.1%}")
                console.print(f"    Retrieval Accuracy: {cat_df['retrieval_accuracy'].mean():.3f}")
                console.print(f"    Avg ROUGE-L F1:     {cat_df['rougeL_f1'].mean():.3f}")
                console.print(f"    Avg Cosine Sim:     {cat_df['cosine_similarity'].mean():.3f}")
        
        # ── Results Table ──
        console.print("\n[bold]Detailed Results Table:[/bold]\n")
        display_cols = [
            "query", "category", "predicted_type", "routing_correct",
            "retrieval_accuracy", "rougeL_f1", "cosine_similarity", "latency_sec"
        ]
        
        # Format for display
        display_df = df[display_cols].copy()
        display_df["query"] = display_df["query"].str[:50]
        display_df = display_df.round(3)
        
        console.print(display_df.to_string(index=False))
    
    def save_results(self, df: pd.DataFrame, path: str = None) -> None:
        """Save evaluation results to CSV."""
        path = path or config.eval_results_path
        df.to_csv(path, index=False)
        console.print(f"\n[green]Results saved to {path}[/green]")


# ============================================================
# Main Entry Point
# ============================================================
def main():
    """Run the full evaluation pipeline."""
    console.print("[bold]Agentic RAG System — Evaluation Framework[/bold]\n")
    
    pipeline = EvaluationPipeline()
    
    # Run evaluation
    df = pipeline.run()
    
    # Print summary
    pipeline.print_summary(df)
    
    # Save to CSV
    pipeline.save_results(df)
    
    # ── Final Verdict ──
    console.rule("[bold]Final Verdict[/bold]")
    routing_acc = df["routing_correct"].mean()
    avg_rouge = df["rougeL_f1"].mean()
    avg_cosine = df["cosine_similarity"].mean()
    
    if routing_acc >= 0.8 and avg_rouge >= 0.2:
        console.print("[bold green]✅ System PASSES evaluation criteria.[/bold green]")
    else:
        console.print("[bold yellow]⚠️ System needs improvement.[/bold yellow]")
        console.print(f"  Routing accuracy: {routing_acc:.1%} (target: ≥80%)")
        console.print(f"  Avg ROUGE-L: {avg_rouge:.3f} (target: ≥0.200)")
    
    return df


if __name__ == "__main__":
    main()
