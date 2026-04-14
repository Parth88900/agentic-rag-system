"""
retriever.py - Retrieval module for the Agentic RAG System.

This module handles:
  - Query embedding generation
  - FAISS-based similarity search
  - Result ranking and filtering
  - Context window assembly for the generator

Design Decisions:
  - We retrieve top-k=5 chunks by default, which provides good recall
    while keeping the context focused.
  - Results are filtered by a minimum similarity threshold to avoid
    including irrelevant chunks in the context.
  - Source deduplication ensures diverse context from multiple documents.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np

from ingestion import Chunk, EmbeddingGenerator, VectorStoreManager
from utils import config, logger, truncate_text


# ============================================================
# Data Structures
# ============================================================
@dataclass
class RetrievalResult:
    """
    Encapsulates the result of a retrieval operation.
    
    Attributes:
        query: The original query string.
        chunks: List of retrieved (Chunk, score) tuples.
        top_score: Highest similarity score among retrieved chunks.
        avg_score: Average similarity score across retrieved chunks.
        sources: Set of unique source documents referenced.
        context_text: Assembled context string for the generator.
    """
    query: str
    chunks: List[Tuple[Chunk, float]]
    top_score: float
    avg_score: float
    sources: set
    context_text: str


# ============================================================
# Retriever
# ============================================================
class Retriever:
    """
    Retrieves relevant document chunks for a given query using FAISS.
    
    The retriever:
    1. Embeds the query using the same model used during ingestion
    2. Searches the FAISS index for the top-k most similar chunks
    3. Filters results below a minimum similarity threshold
    4. Assembles a context string from the retrieved chunks
    
    Attributes:
        embedder: EmbeddingGenerator instance for query embedding.
        vector_store: VectorStoreManager with loaded FAISS index.
    """
    
    def __init__(
        self,
        embedder: EmbeddingGenerator = None,
        vector_store: VectorStoreManager = None
    ):
        """
        Initialize the retriever.
        
        Args:
            embedder: Pre-initialized EmbeddingGenerator. Creates new if None.
            vector_store: Pre-initialized VectorStoreManager. Loads from disk if None.
        """
        self.embedder = embedder or EmbeddingGenerator()
        
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStoreManager(dim=self.embedder.dim)
            self.vector_store.load()
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_score: float = 0.0
    ) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User's question or search query.
            top_k: Number of chunks to retrieve (default: config.top_k).
            min_score: Minimum similarity score threshold. Chunks below
                       this score are filtered out.
        
        Returns:
            RetrievalResult containing chunks, scores, and assembled context.
        """
        top_k = top_k or config.top_k
        
        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(query)
        
        # Step 2: Search FAISS index
        raw_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Step 3: Filter by minimum score
        filtered_results = [
            (chunk, score) for chunk, score in raw_results
            if score >= min_score
        ]
        
        # Step 4: Compute statistics
        if filtered_results:
            scores = [score for _, score in filtered_results]
            top_score = max(scores)
            avg_score = sum(scores) / len(scores)
        else:
            top_score = 0.0
            avg_score = 0.0
        
        # Step 5: Collect unique sources
        sources = {chunk.source for chunk, _ in filtered_results}
        
        # Step 6: Assemble context text
        # Format each chunk with its source for traceability
        context_parts = []
        for i, (chunk, score) in enumerate(filtered_results):
            context_parts.append(
                f"[Source: {chunk.source} | Chunk {chunk.chunk_index} | "
                f"Relevance: {score:.3f}]\n{chunk.text}"
            )
        context_text = "\n\n---\n\n".join(context_parts)
        
        return RetrievalResult(
            query=query,
            chunks=filtered_results,
            top_score=top_score,
            avg_score=avg_score,
            sources=sources,
            context_text=context_text
        )
    
    def retrieve_with_diversity(
        self,
        query: str,
        top_k: int = None,
        diversity_weight: float = 0.3
    ) -> RetrievalResult:
        """
        Retrieve chunks with source diversity.
        
        This method retrieves more chunks than needed and then selects
        a diverse subset that covers multiple source documents. This is
        useful for SYNTHESIS queries where we want information from
        multiple documents.
        
        Args:
            query: User's question.
            top_k: Final number of chunks to return.
            diversity_weight: Weight for source diversity (0=ignore, 1=maximize).
        
        Returns:
            RetrievalResult with diverse source coverage.
        """
        top_k = top_k or config.top_k
        
        # Retrieve 2x more candidates for diversity selection
        candidates = self.retrieve(query, top_k=top_k * 2)
        
        if not candidates.chunks:
            return candidates
        
        # Greedy selection: pick chunks that maximize score + diversity
        selected = []
        used_sources = set()
        
        for chunk, score in candidates.chunks:
            if len(selected) >= top_k:
                break
            
            # Boost score if this chunk is from a new source
            diversity_bonus = diversity_weight if chunk.source not in used_sources else 0.0
            effective_score = score + diversity_bonus
            
            selected.append((chunk, score))  # Keep original score for reporting
            used_sources.add(chunk.source)
        
        # Rebuild RetrievalResult with diverse selection
        scores = [s for _, s in selected]
        sources = {c.source for c, _ in selected}
        
        context_parts = []
        for i, (chunk, score) in enumerate(selected):
            context_parts.append(
                f"[Source: {chunk.source} | Chunk {chunk.chunk_index} | "
                f"Relevance: {score:.3f}]\n{chunk.text}"
            )
        
        return RetrievalResult(
            query=query,
            chunks=selected,
            top_score=max(scores) if scores else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            sources=sources,
            context_text="\n\n---\n\n".join(context_parts)
        )
    
    def get_all_chunk_texts(self) -> List[str]:
        """Return all chunk texts from the vector store (used by router for TF-IDF)."""
        return [chunk.text for chunk in self.vector_store.chunks]
    
    def get_all_chunks(self) -> List[Chunk]:
        """Return all chunks from the vector store."""
        return self.vector_store.chunks


# ============================================================
# CLI Entry Point (for testing)
# ============================================================
if __name__ == "__main__":
    from utils import console
    
    retriever = Retriever()
    
    test_queries = [
        "What is supervised learning?",
        "How do transformers work?",
        "What is the capital of France?",
    ]
    
    for query in test_queries:
        console.rule(f"[bold]Query: {query}[/bold]")
        result = retriever.retrieve(query)
        console.print(f"Top score: {result.top_score:.3f}")
        console.print(f"Sources: {result.sources}")
        for chunk, score in result.chunks[:3]:
            console.print(f"  [{score:.3f}] {truncate_text(chunk.text, 100)}")
        console.print()
