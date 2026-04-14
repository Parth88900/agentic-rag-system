"""
ingestion.py - Document Ingestion Pipeline for the Agentic RAG System.

This module handles the full ingestion flow:
  1. Load documents from a folder (supports .txt, .md, .pdf placeholder)
  2. Clean text (remove formatting artifacts, normalize whitespace)
  3. Chunk documents with configurable size and overlap
  4. Generate embeddings using sentence-transformers (HuggingFace)
  5. Store embeddings in FAISS vector database

Design Decisions:
  - CHUNKING STRATEGY: We use a sliding window approach with 500-char chunks
    and 100-char overlap. This is preferred over sentence-level chunking because:
    (a) Fixed-size chunks provide consistent retrieval granularity
    (b) The overlap ensures no context is lost at chunk boundaries
    (c) 500 chars ≈ 3-4 sentences, which is enough for a coherent answer unit
    
  - EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2 is chosen for:
    (a) Free, no API key required
    (b) Fast inference on CPU (~14K sentences/sec)
    (c) 384-dim vectors keep FAISS index small
    (d) Good quality on semantic similarity benchmarks (STS)
    
  - FAISS INDEX: We use IndexFlatIP (inner product / cosine similarity) for
    exact search. For production with >1M chunks, switch to IndexIVFFlat or
    IndexHNSW for approximate but faster search.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils import config, logger, clean_text, console


# ============================================================
# Data Structures
# ============================================================
@dataclass
class Chunk:
    """
    Represents a text chunk with metadata for traceability.
    
    Attributes:
        text: The chunk's text content.
        source: Original document filename.
        chunk_index: Position of this chunk within the document.
        start_char: Starting character offset in the original document.
        end_char: Ending character offset in the original document.
    """
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    
    def __repr__(self) -> str:
        return f"Chunk(source='{self.source}', idx={self.chunk_index}, len={len(self.text)})"


# ============================================================
# Document Loader
# ============================================================
class DocumentLoader:
    """
    Loads documents from a specified directory.
    
    Supported formats: .txt, .md
    Each document is returned as a dict with 'filename' and 'content' keys.
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".md"}
    
    def __init__(self, docs_dir: str = None):
        self.docs_dir = docs_dir or config.docs_dir
    
    def load(self) -> List[Dict[str, str]]:
        """
        Load all supported documents from the configured directory.
        
        Returns:
            List of dicts, each with 'filename' and 'content' keys.
        
        Raises:
            FileNotFoundError: If docs_dir does not exist.
        """
        docs_path = Path(self.docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")
        
        documents = []
        for filepath in sorted(docs_path.iterdir()):
            if filepath.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    content = filepath.read_text(encoding="utf-8")
                    documents.append({
                        "filename": filepath.name,
                        "content": content
                    })
                    logger.info(f"Loaded: {filepath.name} ({len(content):,} chars)")
                except Exception as e:
                    logger.warning(f"Failed to load {filepath.name}: {e}")
        
        if not documents:
            logger.warning(f"No supported documents found in {self.docs_dir}")
        else:
            logger.info(f"Total documents loaded: {len(documents)}")
        
        return documents


# ============================================================
# Text Chunker
# ============================================================
class TextChunker:
    """
    Splits documents into overlapping chunks using a sliding window approach.
    
    CHUNKING STRATEGY EXPLANATION:
    
    We use character-based chunking with a sliding window rather than
    sentence-based or paragraph-based chunking for these reasons:
    
    1. CONSISTENCY: Fixed-size chunks ensure predictable retrieval behavior.
       Variable-size chunks (e.g., paragraphs) can range from 50 to 2000+ chars,
       leading to inconsistent similarity scores.
    
    2. BOUNDARY HANDLING: The overlap parameter ensures that if a key piece of
       information spans two chunks, it appears in both. With 500-char chunks
       and 100-char overlap, we have ~20% redundancy — a good trade-off between
       coverage and storage.
    
    3. SMART SPLITTING: We try to split at sentence boundaries within the chunk
       window to avoid cutting mid-sentence. This produces more coherent chunks
       while maintaining approximate size consistency.
    
    Parameters:
        chunk_size (int): Target size of each chunk in characters. Default: 500.
            - Too small (< 200): Fragments context, hurts answer quality
            - Too large (> 1000): Dilutes relevant info, hurts retrieval precision
            - Sweet spot: 300-600 chars for most use cases
        
        chunk_overlap (int): Number of overlapping characters between consecutive
            chunks. Default: 100.
            - Should be 10-30% of chunk_size
            - Ensures continuity across chunk boundaries
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        
        # Validate parameters
        assert self.chunk_overlap < self.chunk_size, \
            f"Overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
    
    def _find_split_point(self, text: str, target: int) -> int:
        """
        Find the best split point near `target` index, preferring sentence boundaries.
        
        Looks for sentence-ending punctuation (. ! ?) followed by whitespace
        within a window around the target position.
        
        Args:
            text: The text to split.
            target: Target character index for splitting.
        
        Returns:
            Best split index.
        """
        # Search window: ±50 chars around target
        window = 50
        start = max(0, target - window)
        end = min(len(text), target + window)
        
        # Look for sentence boundaries (. ! ? followed by space/newline)
        best_split = target
        min_distance = float("inf")
        
        for i in range(start, end - 1):
            if text[i] in ".!?\n" and (i + 1 >= len(text) or text[i + 1] in " \n"):
                distance = abs(i + 1 - target)
                if distance < min_distance:
                    min_distance = distance
                    best_split = i + 1
        
        return best_split
    
    def chunk_document(self, text: str, source: str) -> List[Chunk]:
        """
        Split a document into overlapping chunks.
        
        Args:
            text: Cleaned document text.
            source: Source filename for metadata.
        
        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            if end < len(text):
                # Find a good split point near the end
                end = self._find_split_point(text, end)
            else:
                end = len(text)
            
            # Extract chunk text and strip whitespace
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Skip empty chunks
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_idx,
                    start_char=start,
                    end_char=end
                ))
                chunk_idx += 1
            
            if end >= len(text):
                break
            
            # Move start forward by (chunk_size - overlap)
            # This creates the sliding window effect
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of dicts with 'filename' and 'content' keys.
        
        Returns:
            List of all Chunk objects from all documents.
        """
        all_chunks = []
        for doc in documents:
            cleaned = clean_text(doc["content"])
            chunks = self.chunk_document(cleaned, source=doc["filename"])
            all_chunks.extend(chunks)
            logger.info(f"  Chunked {doc['filename']} → {len(chunks)} chunks")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


# ============================================================
# Embedding Generator
# ============================================================
class EmbeddingGenerator:
    """
    Generates dense vector embeddings for text using sentence-transformers.
    
    MODEL CHOICE: all-MiniLM-L6-v2
    - 384 dimensions (compact, fast FAISS search)
    - 80MB model size (quick download)
    - Trained on 1B+ sentence pairs
    - Good performance on STS benchmarks
    - No API key required (runs locally)
    
    For higher quality (at the cost of speed/size), use:
    - 'all-mpnet-base-v2' (768-dim, best quality)
    - 'all-distilroberta-v1' (768-dim, good balance)
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dim}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for efficient encoding.
        
        Returns:
            NumPy array of shape (len(texts), embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True  # L2-normalize for cosine similarity via dot product
        )
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string.
        
        Returns:
            NumPy array of shape (1, embedding_dim).
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return np.array(embedding, dtype=np.float32)


# ============================================================
# Vector Store Manager (FAISS)
# ============================================================
class VectorStoreManager:
    """
    Manages the FAISS vector index for storing and searching embeddings.
    
    INDEX TYPE: IndexFlatIP (Inner Product)
    - Since embeddings are L2-normalized, inner product = cosine similarity
    - Exact search (no approximation) — suitable for <100K vectors
    - For larger datasets, consider:
      - IndexIVFFlat: Inverted file index for faster search (~10x speedup)
      - IndexHNSW: Graph-based index for high recall at speed
    """
    
    def __init__(self, dim: int = None):
        self.dim = dim or config.embedding_dim
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
    
    def build_index(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """
        Build a FAISS index from embeddings and store associated chunk metadata.
        
        Args:
            embeddings: NumPy array of shape (n_chunks, embedding_dim).
            chunks: List of Chunk objects (must be same length as embeddings).
        """
        assert len(embeddings) == len(chunks), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
        
        # Create FAISS index using inner product (cosine similarity since vectors are normalized)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.chunks = chunks
        
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={self.dim}")
    
    def save(self, index_path: str = None, metadata_path: str = None) -> None:
        """Save FAISS index and chunk metadata to disk."""
        index_path = index_path or config.faiss_index_path
        metadata_path = metadata_path or config.metadata_path
        
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Index saved to {index_path}, metadata to {metadata_path}")
    
    def load(self, index_path: str = None, metadata_path: str = None) -> None:
        """Load FAISS index and chunk metadata from disk."""
        index_path = index_path or config.faiss_index_path
        metadata_path = metadata_path or config.metadata_path
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Index or metadata not found. Run ingestion first.\n"
                f"  Expected: {index_path}, {metadata_path}"
            )
        
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.chunks = pickle.load(f)
        
        self.dim = self.index.d
        logger.info(f"Index loaded: {self.index.ntotal} vectors, dim={self.dim}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Tuple[Chunk, float]]:
        """
        Search the FAISS index for the most similar chunks.
        
        Args:
            query_embedding: Query embedding of shape (1, dim).
            top_k: Number of results to return.
        
        Returns:
            List of (Chunk, similarity_score) tuples, sorted by descending similarity.
        """
        top_k = top_k or config.top_k
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not loaded.")
            return []
        
        # Clamp top_k to available vectors
        top_k = min(top_k, self.index.ntotal)
        
        # FAISS search returns distances and indices
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty results
                results.append((self.chunks[idx], float(score)))
        
        return results


# ============================================================
# Ingestion Pipeline Orchestrator
# ============================================================
class IngestionPipeline:
    """
    Orchestrates the full document ingestion pipeline:
    Load → Clean → Chunk → Embed → Store
    """
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStoreManager(dim=self.embedder.dim)
    
    def run(self, save: bool = True) -> VectorStoreManager:
        """
        Execute the full ingestion pipeline.
        
        Args:
            save: Whether to persist the index to disk.
        
        Returns:
            VectorStoreManager with built index.
        """
        console.rule("[bold blue]Document Ingestion Pipeline[/bold blue]")
        
        # Step 1: Load documents
        console.print("\n[bold]Step 1:[/bold] Loading documents...")
        documents = self.loader.load()
        if not documents:
            raise ValueError("No documents to ingest. Check your docs_dir setting.")
        
        # Step 2: Chunk documents (cleaning happens inside chunker)
        console.print("\n[bold]Step 2:[/bold] Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        
        # Step 3: Generate embeddings
        console.print("\n[bold]Step 3:[/bold] Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Step 4: Build FAISS index
        console.print("\n[bold]Step 4:[/bold] Building FAISS index...")
        self.vector_store.build_index(embeddings, chunks)
        
        # Step 5: Save to disk
        if save:
            console.print("\n[bold]Step 5:[/bold] Saving index to disk...")
            self.vector_store.save()
        
        console.rule("[bold green]Ingestion Complete[/bold green]")
        return self.vector_store


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
