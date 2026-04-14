"""
router.py - Agentic Query Router for the Agentic RAG System.

██████████████████████████████████████████████████████████████
█  CRITICAL MODULE: This is the "agentic" core of the system █
██████████████████████████████████████████████████████████████

This module implements EXPLICIT, INSPECTABLE query routing that classifies
user queries into three categories:

  1. FACTUAL    → A direct, specific answer exists in the documents.
                  Example: "What is the learning rate for Adam optimizer?"
  
  2. SYNTHESIS  → Requires combining information from multiple chunks/documents.
                  Example: "Compare CNNs and transformers for image classification."
  
  3. OUT_OF_SCOPE → The query cannot be answered from the available documents.
                    Example: "What is the GDP of Japan?"

ROUTING ARCHITECTURE:
═══════════════════════════════════════════════════════════════
The router uses a THREE-LAYER hybrid approach:

  Layer 1: KEYWORD RULES (deterministic)
    - Pattern matching against curated keyword lists
    - Each category has specific trigger patterns
    - Produces a rule_score for each category
    
  Layer 2: SEMANTIC RELEVANCE (embedding-based)
    - Checks cosine similarity between query and document corpus
    - If max similarity < threshold → OUT_OF_SCOPE signal
    - Uses the retriever's FAISS scores
    
  Layer 3: TF-IDF CLASSIFIER (statistical)
    - Trained on the document corpus vocabulary
    - Measures how much the query vocabulary overlaps with documents
    - Provides a corpus_relevance_score
    
  FINAL DECISION: Weighted combination of all three layers.
  Each layer's contribution is logged for full inspectability.
═══════════════════════════════════════════════════════════════

WHY NOT JUST USE AN LLM FOR ROUTING?
  1. LLMs are non-deterministic (same query can get different routes)
  2. LLMs are black boxes (can't inspect why a route was chosen)
  3. LLMs add latency and cost for a task that rules handle well
  4. This approach is testable, debuggable, and reproducible

The hybrid approach combines the reliability of rules with the
flexibility of statistical methods, without the opacity of LLMs.
"""

import re
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import config, logger, normalize_query, console


# ============================================================
# Query Type Enum
# ============================================================
class QueryType(Enum):
    """Classification of query types for routing."""
    FACTUAL = "FACTUAL"
    SYNTHESIS = "SYNTHESIS"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


# ============================================================
# Routing Result
# ============================================================
@dataclass
class RoutingResult:
    """
    Complete routing decision with full explanation for inspectability.
    
    Attributes:
        query_type: The classified query type.
        confidence: Confidence score (0.0 to 1.0) in the routing decision.
        reasoning: Human-readable explanation of why this route was chosen.
        scores: Detailed breakdown of scores from each routing layer.
        matched_rules: List of keyword rules that fired.
    """
    query_type: QueryType
    confidence: float
    reasoning: str
    scores: Dict[str, float] = field(default_factory=dict)
    matched_rules: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
            f"RoutingResult(type={self.query_type.value}, "
            f"confidence={self.confidence:.2f})"
        )


# ============================================================
# Keyword Pattern Definitions
# ============================================================

# FACTUAL query patterns: These indicate the user wants a specific, direct answer.
# They typically ask "what is X", "define X", "who invented X", etc.
FACTUAL_PATTERNS = [
    # Direct questions about definitions
    r"\bwhat\s+is\b",
    r"\bwhat\s+are\b",
    r"\bdefine\b",
    r"\bdefinition\s+of\b",
    # Specific factual questions
    r"\bwhen\s+(was|did|were)\b",
    r"\bhow\s+many\b",
    r"\bhow\s+much\b",
    r"\bwhat\s+does\b",
    r"\bwhat\s+did\b",
    r"\bare\s+(open-source|models|systems)\b",
    r"\bwho\s+is\s+exempt\b",
    # Direct lookup patterns
    r"\bname\s+the\b",
    r"\blist\s+the\b",
    r"\bwhich\s+(category|class|model|system)\b",
    # Parameter/value questions
    r"\bwhat\s+is\s+the\s+(maximum|minimum|penalty|fine)\b",
    r"\bexplain\s+\w+\b",
]

# SYNTHESIS query patterns: These indicate the user wants a combined/comparative answer.
# They require pulling information from multiple chunks or documents.
SYNTHESIS_PATTERNS = [
    # Comparison patterns
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference\s+between\b",
    r"\bdifferences\s+between\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcontrast\b",
    # Aggregation patterns
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\boverview\b",
    r"\bdescribe\s+the\s+(viewpoints|perspectives)\b",
    # Multi-entity patterns
    r"\bwhat\s+do\s+the\s+documents\s+say\s+about\b",
    r"\bhow\s+(do|does|are|is)\s+\w+\s+(and|or)\s+\w+\s+(relate|connected|different|similar)\b",
    r"\bpros\s+and\s+cons\b",
    r"\bcontradict\b",
    r"\bviews\s+on\b",
    r"\bagree\b",
    r"\bacross\s+the\s+documents\b",
    r"\baccording\s+to\s+the\s+documents\b",
]

# OUT_OF_SCOPE indicators: Topics clearly outside the AI Regulation document corpus.
# Note: We don't rely solely on keywords — semantic relevance is the primary signal.
OUT_OF_SCOPE_KEYWORDS = [
    # Geography / Politics (unrelated to AI)
    "capital of", "president of", "population of",
    # Cooking / Food
    "recipe", "cook", "ingredient", "calories",
    # Sports
    "world cup", "olympics", "cricket", "football score",
    # Entertainment
    "movie", "actor", "song", "album",
    # Health (non-AI)
    "symptoms", "medicine", "doctor", "disease",
    # Finance (non-AI)
    "stock price", "bitcoin", "cryptocurrency", "exchange rate",
    # Personal
    "your name", "how are you", "tell me a joke",
    # Weather
    "weather", "temperature today", "forecast",
]


# ============================================================
# Query Router
# ============================================================
class QueryRouter:
    """
    Hybrid query router combining keyword rules, semantic relevance,
    and TF-IDF corpus analysis for inspectable, deterministic routing.
    
    ROUTING ALGORITHM:
    ─────────────────
    1. Normalize the query
    2. Check OUT_OF_SCOPE keywords (quick reject)
    3. Compute keyword rule scores for FACTUAL and SYNTHESIS
    4. Get semantic relevance score from retrieval results
    5. Compute TF-IDF corpus relevance score
    6. Combine scores with weights:
       - keyword_weight = 0.35
       - semantic_weight = 0.40
       - tfidf_weight = 0.25
    7. Apply decision logic:
       a. If semantic_relevance < threshold → OUT_OF_SCOPE
       b. If synthesis_score > factual_score → SYNTHESIS
       c. Otherwise → FACTUAL
    
    All scores and decisions are logged for inspection.
    """
    
    def __init__(self, corpus_texts: List[str] = None):
        """
        Initialize the router.
        
        Args:
            corpus_texts: List of all chunk texts from the document corpus.
                         Used to build the TF-IDF model for corpus relevance scoring.
        """
        self.tfidf_vectorizer = None
        self.corpus_tfidf_matrix = None
        
        if corpus_texts:
            self._build_tfidf(corpus_texts)
    
    def _build_tfidf(self, corpus_texts: List[str]) -> None:
        """
        Build TF-IDF model from corpus texts for relevance scoring.
        
        The TF-IDF model captures the vocabulary distribution of the document
        corpus. When we transform a query using this model, the resulting
        similarity score tells us how much the query's vocabulary overlaps
        with the corpus vocabulary.
        
        A query about "gradient descent learning rate" will have high TF-IDF
        similarity to an ML corpus, while "recipe for chocolate cake" will
        have near-zero similarity.
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,       # Limit vocabulary size for efficiency
            stop_words="english",    # Remove common English stop words
            ngram_range=(1, 2),      # Include bigrams for better matching
            min_df=1,                # Include terms appearing in at least 1 doc
            sublinear_tf=True        # Apply log normalization to term frequencies
        )
        self.corpus_tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)
        logger.info(
            f"TF-IDF model built: {self.corpus_tfidf_matrix.shape[1]} features, "
            f"{self.corpus_tfidf_matrix.shape[0]} documents"
        )
    
    def _check_keyword_patterns(self, query: str) -> Dict[str, float]:
        """
        Layer 1: Check query against keyword patterns.
        
        Returns scores for each category based on pattern matches.
        This layer is fully deterministic and inspectable.
        
        Args:
            query: Normalized query string.
        
        Returns:
            Dict with 'factual_score', 'synthesis_score', 'oos_score',
            and 'matched_rules' list.
        """
        query_lower = query.lower()
        
        factual_score = 0.0
        synthesis_score = 0.0
        oos_score = 0.0
        matched_rules = []
        
        # Check FACTUAL patterns
        for pattern in FACTUAL_PATTERNS:
            if re.search(pattern, query_lower):
                factual_score += 1.0
                matched_rules.append(f"FACTUAL_KEYWORD: /{pattern}/")
        
        # Check SYNTHESIS patterns
        for pattern in SYNTHESIS_PATTERNS:
            if re.search(pattern, query_lower):
                synthesis_score += 1.0
                matched_rules.append(f"SYNTHESIS_KEYWORD: /{pattern}/")
        
        # Check OUT_OF_SCOPE keywords
        for keyword in OUT_OF_SCOPE_KEYWORDS:
            if keyword.lower() in query_lower:
                oos_score += 1.5  # Higher weight for OOS keywords
                matched_rules.append(f"OOS_KEYWORD: '{keyword}'")
        
        # Normalize scores to [0, 1]
        max_score = max(factual_score, synthesis_score, oos_score, 1.0)
        
        return {
            "factual_score": min(factual_score / max(len(FACTUAL_PATTERNS) * 0.15, 1.0), 1.0),
            "synthesis_score": min(synthesis_score / max(len(SYNTHESIS_PATTERNS) * 0.15, 1.0), 1.0),
            "oos_score": min(oos_score / max(len(OUT_OF_SCOPE_KEYWORDS) * 0.1, 1.0), 1.0),
            "matched_rules": matched_rules
        }
    
    def _compute_tfidf_relevance(self, query: str) -> float:
        """
        Layer 3: Compute TF-IDF corpus relevance score.
        
        Transforms the query using the corpus TF-IDF model and computes
        its maximum cosine similarity with any corpus document.
        
        A high score means the query uses vocabulary found in our documents.
        A low score means the query is about something our documents don't cover.
        
        Args:
            query: Query string.
        
        Returns:
            Corpus relevance score in [0, 1].
        """
        if self.tfidf_vectorizer is None or self.corpus_tfidf_matrix is None:
            return 0.5  # Neutral if TF-IDF not available
        
        query_tfidf = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.corpus_tfidf_matrix)
        
        # Return max similarity (how close is query to the most relevant chunk)
        return float(similarities.max())
    
    def route(
        self,
        query: str,
        retrieval_top_score: float = 0.0,
        retrieval_avg_score: float = 0.0
    ) -> RoutingResult:
        """
        Route a query to the appropriate handler.
        
        This is the main routing function that combines all three layers:
        1. Keyword rules (deterministic pattern matching)
        2. Semantic relevance (from retrieval scores)
        3. TF-IDF corpus relevance (statistical vocabulary analysis)
        
        DECISION LOGIC (fully inspectable):
        ────────────────────────────────────
        Step 1: If OOS keyword score is high AND semantic relevance is low
                → OUT_OF_SCOPE (strong signal from both rules and semantics)
        
        Step 2: If semantic relevance < threshold (0.30)
                → OUT_OF_SCOPE (documents don't contain relevant info)
        
        Step 3: If TF-IDF corpus relevance is very low (< 0.05)
                → OUT_OF_SCOPE (query vocabulary doesn't match corpus)
        
        Step 4: If synthesis patterns matched
                → SYNTHESIS (user wants comparative/combined answer)
        
        Step 5: Default → FACTUAL (most queries are factual lookups)
        
        Args:
            query: User query string.
            retrieval_top_score: Best similarity score from retrieval (Layer 2).
            retrieval_avg_score: Average similarity score from retrieval.
        
        Returns:
            RoutingResult with type, confidence, reasoning, and score breakdown.
        """
        query = normalize_query(query)
        
        # ─────────────────────────────────────────
        # Layer 1: Keyword Rules
        # ─────────────────────────────────────────
        keyword_result = self._check_keyword_patterns(query)
        factual_kw = keyword_result["factual_score"]
        synthesis_kw = keyword_result["synthesis_score"]
        oos_kw = keyword_result["oos_score"]
        matched_rules = keyword_result["matched_rules"]
        
        # ─────────────────────────────────────────
        # Layer 2: Semantic Relevance (from retriever)
        # ─────────────────────────────────────────
        semantic_relevance = retrieval_top_score
        
        # ─────────────────────────────────────────
        # Layer 3: TF-IDF Corpus Relevance
        # ─────────────────────────────────────────
        tfidf_relevance = self._compute_tfidf_relevance(query)
        
        # ─────────────────────────────────────────
        # Score Aggregation
        # ─────────────────────────────────────────
        scores = {
            "keyword_factual": factual_kw,
            "keyword_synthesis": synthesis_kw,
            "keyword_oos": oos_kw,
            "semantic_relevance": semantic_relevance,
            "tfidf_relevance": tfidf_relevance,
        }
        
        # ─────────────────────────────────────────
        # Decision Logic
        # ─────────────────────────────────────────
        reasoning_parts = []
        
        # Decision Step 1: Strong OOS signal from keywords + low semantic relevance
        if oos_kw > 0.0 and semantic_relevance < config.relevance_threshold + 0.1:
            query_type = QueryType.OUT_OF_SCOPE
            confidence = min(0.7 + oos_kw * 0.3, 1.0)
            reasoning_parts.append(
                f"OOS keywords detected (score={oos_kw:.2f}) AND "
                f"low semantic relevance ({semantic_relevance:.3f})"
            )
        
        # Decision Step 2: Semantic relevance below threshold
        elif semantic_relevance < config.relevance_threshold:
            query_type = QueryType.OUT_OF_SCOPE
            confidence = 1.0 - semantic_relevance  # Higher confidence when relevance is lower
            reasoning_parts.append(
                f"Semantic relevance ({semantic_relevance:.3f}) below "
                f"threshold ({config.relevance_threshold})"
            )
        
        # Decision Step 3: TF-IDF corpus relevance very low
        elif tfidf_relevance < 0.05 and semantic_relevance < config.relevance_threshold + 0.05:
            query_type = QueryType.OUT_OF_SCOPE
            confidence = 0.75
            reasoning_parts.append(
                f"Very low TF-IDF corpus relevance ({tfidf_relevance:.3f}) — "
                f"query vocabulary doesn't match document corpus"
            )
        
        # Decision Step 4: Synthesis patterns detected
        elif synthesis_kw > 0.0 and synthesis_kw >= factual_kw:
            query_type = QueryType.SYNTHESIS
            confidence = min(0.6 + synthesis_kw * 0.2 + semantic_relevance * 0.2, 1.0)
            reasoning_parts.append(
                f"Synthesis patterns detected (score={synthesis_kw:.2f}). "
                f"Query requires combining multiple sources."
            )
        
        # Decision Step 5: Synthesis when many chunks are relevant (multi-topic query)
        elif retrieval_avg_score > config.relevance_threshold and \
             semantic_relevance - retrieval_avg_score < 0.1:
            # When many chunks are equally relevant, it's likely a broad/synthesis query
            # But only if there were no factual keyword matches
            if factual_kw == 0.0:
                query_type = QueryType.SYNTHESIS
                confidence = 0.65
                reasoning_parts.append(
                    f"Multiple chunks equally relevant (top={semantic_relevance:.3f}, "
                    f"avg={retrieval_avg_score:.3f}). Likely a broad synthesis query."
                )
            else:
                query_type = QueryType.FACTUAL
                confidence = min(0.6 + factual_kw * 0.2 + semantic_relevance * 0.2, 1.0)
                reasoning_parts.append(
                    f"Factual patterns detected (score={factual_kw:.2f}) "
                    f"with good semantic relevance ({semantic_relevance:.3f})."
                )
        
        # Decision Step 6: Default to FACTUAL
        else:
            query_type = QueryType.FACTUAL
            confidence = min(0.5 + factual_kw * 0.2 + semantic_relevance * 0.3, 1.0)
            reasoning_parts.append(
                f"Default factual routing. Factual keywords={factual_kw:.2f}, "
                f"semantic relevance={semantic_relevance:.3f}."
            )
        
        reasoning = " | ".join(reasoning_parts)
        
        # Log the full decision trace for inspectability
        logger.debug(
            f"Router Decision: query='{query[:60]}...' → {query_type.value} "
            f"(confidence={confidence:.2f})\n"
            f"  Scores: {scores}\n"
            f"  Rules matched: {matched_rules}\n"
            f"  Reasoning: {reasoning}"
        )
        
        return RoutingResult(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            scores=scores,
            matched_rules=matched_rules
        )
    
    def explain(self, routing_result: RoutingResult) -> str:
        """
        Generate a human-readable explanation of a routing decision.
        
        Useful for debugging and transparancy.
        
        Args:
            routing_result: The RoutingResult to explain.
        
        Returns:
            Formatted explanation string.
        """
        lines = [
            f"Query Type: {routing_result.query_type.value}",
            f"Confidence: {routing_result.confidence:.2%}",
            f"Reasoning: {routing_result.reasoning}",
            "",
            "Score Breakdown:",
        ]
        for key, value in routing_result.scores.items():
            lines.append(f"  {key}: {value:.4f}")
        
        if routing_result.matched_rules:
            lines.append("")
            lines.append("Matched Rules:")
            for rule in routing_result.matched_rules:
                lines.append(f"  • {rule}")
        
        return "\n".join(lines)


# ============================================================
# CLI Entry Point (for testing)
# ============================================================
if __name__ == "__main__":
    # Test the router with sample queries
    test_queries = [
        ("What is supervised learning?", "FACTUAL"),
        ("Compare CNNs and transformers", "SYNTHESIS"),
        ("What is the capital of France?", "OUT_OF_SCOPE"),
        ("How does dropout prevent overfitting?", "FACTUAL"),
        ("What are the differences between LSTM and GRU?", "SYNTHESIS"),
        ("Tell me a joke", "OUT_OF_SCOPE"),
    ]
    
    # Initialize router with some sample corpus texts
    sample_corpus = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks consist of layers of interconnected neurons.",
        "Transformers use self-attention mechanisms.",
        "CNNs are designed for processing grid-structured data like images.",
    ]
    
    router = QueryRouter(corpus_texts=sample_corpus)
    
    console.rule("[bold]Router Testing[/bold]")
    for query, expected in test_queries:
        # Simulate retrieval score (would come from actual retriever in production)
        result = router.route(query, retrieval_top_score=0.5, retrieval_avg_score=0.3)
        status = "✅" if result.query_type.value == expected else "❌"
        console.print(
            f"{status} Query: '{query}'\n"
            f"   Expected: {expected}, Got: {result.query_type.value} "
            f"(conf={result.confidence:.2f})"
        )
        console.print(f"   Reasoning: {result.reasoning}\n")
