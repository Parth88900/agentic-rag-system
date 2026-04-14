"""
main.py - Entry point for the Agentic RAG System.

This script provides a CLI interface for:
  1. Ingesting documents (building the vector index)
  2. Querying the system interactively
  3. Running the evaluation pipeline

Usage:
  python main.py ingest        # Ingest documents and build index
  python main.py query         # Interactive query mode
  python main.py evaluate      # Run evaluation pipeline
  python main.py demo          # Run a quick demo with sample queries

Architecture Flow:
  ┌────────────┐    ┌──────────┐    ┌────────────┐    ┌───────────┐
  │   Query    │───>│ Retriever│───>│   Router   │───>│ Generator │
  │  (User)    │    │ (FAISS)  │    │ (Hybrid)   │    │(Grounded) │
  └────────────┘    └──────────┘    └────────────┘    └───────────┘
                         │                │                  │
                    Top-k chunks    FACTUAL/SYNTH/    Grounded Answer
                    + scores         OUT_OF_SCOPE     + Confidence
"""

import sys
import os

from utils import config, logger, console, normalize_query
from ingestion import IngestionPipeline, EmbeddingGenerator, VectorStoreManager
from retriever import Retriever
from router import QueryRouter, QueryType
from generator import AnswerGenerator


# ============================================================
# RAG System (Main Pipeline)
# ============================================================
class AgenticRAGSystem:
    """
    The complete Agentic RAG pipeline.
    
    Orchestrates: Retriever → Router → Generator
    """
    
    def __init__(self):
        """Initialize all components."""
        console.print("[bold]Initializing Agentic RAG System...[/bold]")
        
        # Load embedding model (shared across components)
        self.embedder = EmbeddingGenerator()
        
        # Load vector store
        self.vector_store = VectorStoreManager(dim=self.embedder.dim)
        
        if os.path.exists(config.faiss_index_path) and os.path.exists(config.metadata_path):
            self.vector_store.load()
        else:
            console.print("[yellow]No index found. Running ingestion first...[/yellow]")
            pipeline = IngestionPipeline()
            vs = pipeline.run(save=True)
            self.vector_store = vs
        
        # Initialize components
        self.retriever = Retriever(embedder=self.embedder, vector_store=self.vector_store)
        self.router = QueryRouter(corpus_texts=self.retriever.get_all_chunk_texts())
        self.generator = AnswerGenerator()
        
        console.print("[bold green]System ready![/bold green]\n")
    
    def answer(self, query: str, verbose: bool = True) -> dict:
        """
        Process a query through the full RAG pipeline.
        
        Pipeline:
        1. Retrieve relevant chunks from FAISS
        2. Route the query (classify type)
        3. Generate a grounded answer
        
        Args:
            query: User's question.
            verbose: Print detailed output.
        
        Returns:
            Dict with answer, routing info, retrieval info, and confidence.
        """
        query = normalize_query(query)
        
        # ── Step 1: Retrieve ──
        retrieval_result = self.retriever.retrieve(query)
        
        # ── Step 2: Route ──
        routing_result = self.router.route(
            query,
            retrieval_top_score=retrieval_result.top_score,
            retrieval_avg_score=retrieval_result.avg_score
        )
        
        # For SYNTHESIS queries, re-retrieve with diversity
        if routing_result.query_type == QueryType.SYNTHESIS:
            retrieval_result = self.retriever.retrieve_with_diversity(query)
        
        # ── Step 3: Generate ──
        answer = self.generator.generate(query, routing_result, retrieval_result)
        
        # ── Output ──
        result = {
            "query": query,
            "answer": answer.answer,
            "query_type": routing_result.query_type.value,
            "routing_confidence": routing_result.confidence,
            "answer_confidence": answer.confidence,
            "sources": answer.sources,
            "retrieval_top_score": retrieval_result.top_score,
            "generation_method": answer.generation_method,
            "routing_reasoning": routing_result.reasoning,
        }
        
        if verbose:
            self._print_result(result, routing_result)
        
        return result
    
    def _print_result(self, result: dict, routing_result) -> None:
        """Pretty-print a query result."""
        console.print()
        console.rule(f"[bold]Query: {result['query']}[/bold]")
        
        # Routing info
        type_colors = {
            "FACTUAL": "green",
            "SYNTHESIS": "blue",
            "OUT_OF_SCOPE": "red"
        }
        color = type_colors.get(result["query_type"], "white")
        console.print(f"[{color}]■ Query Type: {result['query_type']}[/{color}] "
                      f"(confidence: {result['routing_confidence']:.0%})")
        console.print(f"  Routing: {result['routing_reasoning']}")
        
        # Answer
        console.print(f"\n[bold]Answer:[/bold]")
        console.print(f"  {result['answer']}")
        
        # Metadata
        console.print(f"\n[dim]Sources: {', '.join(result['sources']) if result['sources'] else 'N/A'}[/dim]")
        console.print(f"[dim]Retrieval Score: {result['retrieval_top_score']:.3f} | "
                      f"Method: {result['generation_method']} | "
                      f"Answer Confidence: {result['answer_confidence']:.0%}[/dim]")
        console.print()


# ============================================================
# CLI Commands
# ============================================================
def cmd_ingest():
    """Run the document ingestion pipeline."""
    pipeline = IngestionPipeline()
    pipeline.run(save=True)


def cmd_query():
    """Interactive query mode."""
    system = AgenticRAGSystem()
    
    console.print("[bold]Interactive Query Mode[/bold]")
    console.print("Type your question and press Enter. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        system.answer(query, verbose=True)


def cmd_evaluate():
    """Run the evaluation pipeline."""
    from evaluation import main as eval_main
    eval_main()


def cmd_demo():
    """Run a demo with sample queries."""
    system = AgenticRAGSystem()
    
    demo_queries = [
        # Factual
        "What is the F1 score metric?",
        "What activation function does BERT use?",
        # Synthesis
        "Compare batch prediction and real-time serving for model deployment.",
        # Out-of-scope
        "What is the weather in New York today?",
    ]
    
    console.rule("[bold blue]Agentic RAG System — Demo[/bold blue]")
    
    for query in demo_queries:
        system.answer(query, verbose=True)


# ============================================================
# Entry Point
# ============================================================
def main():
    """Main CLI dispatcher."""
    console.print("""
[bold blue]
    ╔═══════════════════════════════════════════╗
    ║     Agentic RAG System v1.0               ║
    ║     with Evaluation Framework              ║
    ╚═══════════════════════════════════════════╝
[/bold blue]
    """)
    
    if len(sys.argv) < 2:
        console.print("[bold]Usage:[/bold]")
        console.print("  python main.py ingest     — Ingest documents and build index")
        console.print("  python main.py query      — Interactive query mode")
        console.print("  python main.py evaluate   — Run evaluation pipeline")
        console.print("  python main.py demo       — Quick demo with sample queries")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "ingest":
        cmd_ingest()
    elif command == "query":
        cmd_query()
    elif command in ("evaluate", "eval"):
        cmd_evaluate()
    elif command == "demo":
        cmd_demo()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Valid commands: ingest, query, evaluate, demo")
        sys.exit(1)


if __name__ == "__main__":
    main()
