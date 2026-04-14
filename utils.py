"""
utils.py - Shared utilities and configuration for the Agentic RAG System.

This module provides:
  - Centralized configuration management (paths, model names, thresholds)
  - Text cleaning and normalization functions
  - Logging setup with Rich for colored terminal output
  - Common helper functions used across all modules

Design Decision: We use a dataclass-based Config rather than a YAML/JSON file
to keep the project self-contained and avoid external config file dependencies.
All tuneable parameters are documented with their rationale.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from rich.logging import RichHandler
from rich.console import Console

# Load environment variables from .env file (if present)
load_dotenv()

# ============================================================
# Rich Console for pretty terminal output
# ============================================================
console = Console()


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    """
    Centralized configuration for the entire RAG pipeline.
    
    All parameters have sensible defaults and are documented with
    rationale for the chosen values.
    """
    
    # --- Paths ---
    docs_dir: str = "sample_docs"               # Directory containing source documents
    faiss_index_path: str = "vector_store.index" # Where to save/load FAISS index
    metadata_path: str = "chunk_metadata.pkl"    # Chunk metadata (text + source info)
    
    # --- Embedding Model ---
    # Using sentence-transformers 'all-MiniLM-L6-v2' — a lightweight (80MB) model
    # that produces 384-dimensional embeddings. Good balance of speed and quality.
    # For production, consider 'all-mpnet-base-v2' (768-dim, higher quality).
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # --- Chunking Parameters ---
    # Chunk size of 1500 chars (approx 350-400 tokens): Large enough to preserve context,
    # as required by the prompt (300-500 tokens).
    # Overlap of 300 chars (approx 75 tokens): Required by prompt constraints (50-100 tokens),
    # ensuring continuity across boundaries.
    chunk_size: int = 1500
    chunk_overlap: int = 300
    
    # --- Retrieval Parameters ---
    top_k: int = 5  # Number of chunks to retrieve. 5 provides good recall without noise.
    
    # --- Router Thresholds ---
    relevance_threshold: float = 0.30
    synthesis_keyword_boost: float = 0.15
    
    # --- Generator ---
    # Gemini API key (optional). If not set, falls back to extractive generation.
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = "gemini-2.0-flash"
    max_answer_tokens: int = 500
    temperature: float = 0.1  # Low temperature for factual, grounded answers
    
    # --- Evaluation ---
    eval_results_path: str = "evaluation_results.csv"


# Singleton config instance
config = Config()


# ============================================================
# Logging Setup
# ============================================================
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with Rich handler for colored, formatted output.
    
    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False
        )]
    )
    logger = logging.getLogger("agentic_rag")
    logger.setLevel(level)
    return logger


# Create default logger
logger = setup_logging()


# ============================================================
# Text Cleaning Utilities
# ============================================================
def clean_text(text: str) -> str:
    """
    Clean and normalize raw document text.
    
    Operations performed (in order):
    1. Replace tabs with spaces (formatting normalization)
    2. Collapse multiple spaces into single space
    3. Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
    4. Strip leading/trailing whitespace
    5. Remove non-printable characters (except newlines)
    
    Args:
        text: Raw document text.
    
    Returns:
        Cleaned text string.
    """
    # Step 1: Replace tabs
    text = text.replace("\t", " ")
    
    # Step 2: Collapse multiple spaces (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    
    # Step 3: Collapse excessive newlines (3+ → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Step 4: Strip whitespace
    text = text.strip()
    
    # Step 5: Remove non-printable characters (keep \n, \r, \t and printable)
    text = re.sub(r"[^\x20-\x7E\n\r]", "", text)
    
    return text


def normalize_query(query: str) -> str:
    """
    Normalize a user query for consistent processing.
    
    Args:
        query: Raw user query.
    
    Returns:
        Normalized query string.
    """
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    return query


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
