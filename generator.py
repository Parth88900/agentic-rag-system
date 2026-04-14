"""
generator.py - Answer Generation module for the Agentic RAG System.

This module generates grounded answers based on retrieved context and query type.

GENERATION STRATEGY:
═══════════════════
The generator operates differently based on the routing decision:

  FACTUAL → Extract the most relevant information from the top chunk(s).
            Keep answer concise and directly grounded in the source text.
            
  SYNTHESIS → Combine information from multiple chunks/sources.
              Identify commonalities and differences across sources.
              Produce a structured, comprehensive answer.
              
  OUT_OF_SCOPE → Return a fixed refusal message. NO generation attempted.
                  This is the primary anti-hallucination safeguard.

TWO GENERATION BACKENDS:
  1. Google Gemini API (if GEMINI_API_KEY is set): Uses Gemini with strict
     system prompts that enforce grounding.
  2. Extractive Fallback (no API needed): Selects and combines the most
     relevant sentences from retrieved chunks. Always grounded by design.

ANTI-HALLUCINATION MEASURES:
  - OUT_OF_SCOPE queries get a fixed response (no LLM involved)
  - System prompt explicitly forbids making up information
  - Temperature set to 0.1 (near-deterministic)
  - Extractive fallback cannot hallucinate by design
  - Confidence scores indicate answer reliability
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass

import numpy as np

from router import QueryType, RoutingResult
from retriever import RetrievalResult
from utils import config, logger, console


# ============================================================
# Data Structures
# ============================================================
@dataclass
class GeneratedAnswer:
    """
    Complete answer with metadata for evaluation.
    
    Attributes:
        answer: The generated answer text.
        query_type: How the query was classified.
        confidence: Answer confidence score (0.0 to 1.0).
        sources: List of source documents used.
        generation_method: 'gemini', 'extractive', or 'refusal'.
        reasoning: Explanation of how the answer was generated.
    """
    answer: str
    query_type: QueryType
    confidence: float
    sources: List[str]
    generation_method: str
    reasoning: str


# ============================================================
# Fixed Response for Out-of-Scope Queries
# ============================================================
OUT_OF_SCOPE_RESPONSE = (
    "The provided documents do not contain sufficient information "
    "to answer this question."
)


# ============================================================
# System Prompts for Gemini
# ============================================================
FACTUAL_SYSTEM_PROMPT = """You are a precise, factual question-answering assistant.
Your task is to answer the user's question based ONLY on the provided context.

STRICT RULES:
1. ONLY use information from the provided context passages.
2. Do NOT add any information beyond what is in the context.
3. If the context doesn't fully answer the question, say what you CAN answer and note what's missing.
4. Keep answers concise and direct.
5. Cite the source document when possible.
6. Start with a confidence indicator: [HIGH CONFIDENCE] or [MODERATE CONFIDENCE]."""

SYNTHESIS_SYSTEM_PROMPT = """You are an analytical assistant that synthesizes information from multiple sources.
Your task is to combine information from the provided context passages to answer the user's question.

STRICT RULES:
1. ONLY use information from the provided context passages.
2. Do NOT add any information beyond what is in the context.
3. When comparing topics, clearly identify which source each piece of information comes from.
4. Structure your answer with clear sections if comparing multiple items.
5. EXPLICITLY HIGHLIGHT CONTRADICTIONS: If sources contradict each other, you MUST point out the difference (e.g., 'Document A states X, BUT Document B states Y'). Do NOT blindly merge conflicting info.
6. Start with a confidence indicator: [HIGH CONFIDENCE] or [MODERATE CONFIDENCE]."""


# ============================================================
# Answer Generator
# ============================================================
class AnswerGenerator:
    """
    Generates grounded answers based on retrieved context and query routing.
    
    The generator ensures answers are strictly grounded in the provided
    documents. It uses different strategies based on query type and
    falls back to extractive generation when no API key is available.
    """
    
    def __init__(self):
        """Initialize the generator. Checks for Gemini API key availability."""
        self.gemini_client = None
        
        if config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(config.gemini_model)
                self._genai = genai
                logger.info(f"Gemini client initialized for generation: {config.gemini_model}.")
            except ImportError:
                logger.warning("google-generativeai package not installed. Using extractive fallback.")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}. Using extractive fallback.")
        else:
            logger.info(
                "No GEMINI_API_KEY found. Using extractive generation (fully local, no hallucination)."
            )
    
    def generate(
        self,
        query: str,
        routing_result: RoutingResult,
        retrieval_result: RetrievalResult
    ) -> GeneratedAnswer:
        """
        Generate an answer based on query type and retrieved context.
        
        This is the main entry point. It dispatches to the appropriate
        handler based on the routing decision.
        
        Args:
            query: Original user query.
            routing_result: Classification from the router.
            retrieval_result: Retrieved chunks from the retriever.
        
        Returns:
            GeneratedAnswer with the response and metadata.
        """
        query_type = routing_result.query_type
        
        # ── OUT_OF_SCOPE: Fixed refusal (anti-hallucination) ──
        if query_type == QueryType.OUT_OF_SCOPE:
            return GeneratedAnswer(
                answer=OUT_OF_SCOPE_RESPONSE,
                query_type=query_type,
                confidence=routing_result.confidence,
                sources=[],
                generation_method="refusal",
                reasoning="Query classified as OUT_OF_SCOPE. Returning fixed refusal to prevent hallucination."
            )
        
        # ── FACTUAL / SYNTHESIS: Generate grounded answer ──
        if self.gemini_client:
            return self._generate_gemini(query, query_type, retrieval_result, routing_result)
        else:
            return self._generate_extractive(query, query_type, retrieval_result, routing_result)
    
    def _generate_gemini(
        self,
        query: str,
        query_type: QueryType,
        retrieval_result: RetrievalResult,
        routing_result: RoutingResult
    ) -> GeneratedAnswer:
        """
        Generate answer using Gemini API with strict grounding prompts.
        
        The system prompt enforces that the model only uses provided context.
        Temperature is set very low (0.1) for factual accuracy.
        """
        # Select system prompt based on query type
        system_prompt = (
            FACTUAL_SYSTEM_PROMPT if query_type == QueryType.FACTUAL
            else SYNTHESIS_SYSTEM_PROMPT
        )
        
        # Build user message with context
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Context passages:\n\n{retrieval_result.context_text}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Answer based ONLY on the above context:"
        )
        
        try:
            response = self.gemini_client.generate_content(
                full_prompt,
                generation_config=self._genai.types.GenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_answer_tokens,
                ),
            )
            
            answer_text = response.text.strip()
            
            # --- CONTRADICTION DETECTION MODULE ---
            contradiction_warning = self._detect_contradictions(retrieval_result)
            if contradiction_warning:
                answer_text += contradiction_warning
            
            # Compute confidence based on retrieval scores and routing confidence
            confidence = min(
                routing_result.confidence * 0.4 +
                retrieval_result.top_score * 0.4 +
                (retrieval_result.avg_score * 0.2),
                1.0
            )
            
            return GeneratedAnswer(
                answer=answer_text,
                query_type=query_type,
                confidence=confidence,
                sources=list(retrieval_result.sources),
                generation_method="gemini",
                reasoning=f"Generated via Gemini {config.gemini_model} with grounded prompt."
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}. Falling back to extractive.")
            return self._generate_extractive(query, query_type, retrieval_result, routing_result)
    
    def _generate_extractive(
        self,
        query: str,
        query_type: QueryType,
        retrieval_result: RetrievalResult,
        routing_result: RoutingResult
    ) -> GeneratedAnswer:
        """
        Generate answer by extracting and combining relevant sentences.
        
        This is the fallback method that requires no API key. It works by:
        1. Breaking retrieved chunks into sentences
        2. Scoring sentences by keyword overlap with the query
        3. Selecting the top-scoring sentences
        4. Assembling them into a coherent answer
        
        ADVANTAGE: This method CANNOT hallucinate because it only uses
        text directly from the source documents.
        
        For FACTUAL queries: Select 2-3 most relevant sentences.
        For SYNTHESIS queries: Select 4-6 sentences from multiple sources.
        """
        if not retrieval_result.chunks:
            return GeneratedAnswer(
                answer=OUT_OF_SCOPE_RESPONSE,
                query_type=query_type,
                confidence=0.3,
                sources=[],
                generation_method="extractive",
                reasoning="No relevant chunks retrieved."
            )
        
        # Step 1: Extract all sentences from retrieved chunks
        all_sentences = []
        for chunk, score in retrieval_result.chunks:
            sentences = self._split_sentences(chunk.text)
            for sent in sentences:
                if len(sent.strip()) > 20:  # Filter out very short fragments
                    all_sentences.append({
                        "text": sent.strip(),
                        "source": chunk.source,
                        "chunk_score": score
                    })
        
        if not all_sentences:
            return GeneratedAnswer(
                answer=OUT_OF_SCOPE_RESPONSE,
                query_type=query_type,
                confidence=0.3,
                sources=[],
                generation_method="extractive",
                reasoning="No meaningful sentences found in retrieved chunks."
            )
        
        # Step 2: Score sentences by relevance to query
        query_words = set(query.lower().split())
        # Remove common words for better matching
        stop_words = {"what", "is", "the", "a", "an", "how", "does", "do", "are", "was",
                      "were", "in", "of", "and", "or", "to", "for", "with", "can", "this",
                      "that", "it", "be", "on", "at", "by", "from"}
        query_keywords = query_words - stop_words
        
        for sent_info in all_sentences:
            sent_words = set(sent_info["text"].lower().split())
            # Keyword overlap score
            overlap = len(query_keywords & sent_words)
            # Combined score: chunk relevance + keyword overlap
            sent_info["relevance"] = sent_info["chunk_score"] * 0.6 + (overlap / max(len(query_keywords), 1)) * 0.4
        
        # Step 3: Sort by relevance
        all_sentences.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Step 4: Select sentences based on query type
        if query_type == QueryType.FACTUAL:
            # For factual: take top 2-3 sentences
            selected = all_sentences[:3]
        else:
            # For synthesis: take top 4-6 sentences, prefer diverse sources
            selected = self._select_diverse_sentences(all_sentences, max_sentences=5)
        
        # Step 5: Assemble answer
        answer_parts = []
        sources = set()
        
        if query_type == QueryType.SYNTHESIS:
            # Group by source for synthesis answers
            source_groups = {}
            for sent in selected:
                source = sent["source"]
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(sent["text"])
                sources.add(source)
            
            for source, sents in source_groups.items():
                answer_parts.append(f"From {source}: " + " ".join(sents))
            
            answer_text = "\n\n".join(answer_parts)
        else:
            # For factual: simple concatenation
            for sent in selected:
                answer_parts.append(sent["text"])
                sources.add(sent["source"])
            
            answer_text = " ".join(answer_parts)
        
        # Step 6: Compute confidence
        if selected:
            avg_relevance = sum(s["relevance"] for s in selected) / len(selected)
            confidence = min(avg_relevance + 0.1, 1.0)
        else:
            confidence = 0.3
        
        # Prepend confidence indicator
        conf_label = "[HIGH CONFIDENCE]" if confidence > 0.6 else "[MODERATE CONFIDENCE]"
        answer_text = f"{conf_label} {answer_text}"
        
        # --- CONTRADICTION DETECTION MODULE ---
        contradiction_warning = self._detect_contradictions(retrieval_result)
        if contradiction_warning:
            answer_text += contradiction_warning
        
        return GeneratedAnswer(
            answer=answer_text,
            query_type=query_type,
            confidence=confidence,
            sources=list(sources),
            generation_method="extractive",
            reasoning=(
                f"Extractive generation: selected {len(selected)} sentences "
                f"from {len(sources)} sources. Average relevance: {avg_relevance:.3f}."
                if selected else "No relevant sentences found."
            )
        )
    
    def _detect_contradictions(self, retrieval_result: RetrievalResult) -> str:
        """
        Cross-reference chunks to identify potential contradictions or conflicting information.
        Returns a formatted warning string if a discrepancy is found, else an empty string.
        """
        if len(retrieval_result.sources) < 2:
            return ""
            
        # Group text by source
        source_texts = {}
        for chunk, _ in retrieval_result.chunks:
            if chunk.source not in source_texts:
                source_texts[chunk.source] = []
            source_texts[chunk.source].append(chunk.text)
            
        sources = list(source_texts.keys())
        for i in range(len(sources)):
            for j in range(i+1, len(sources)):
                t1 = " ".join(source_texts[sources[i]]).lower()
                t2 = " ".join(source_texts[sources[j]]).lower()
                
                # Basic heuristic: look for similar numerical contexts with different values
                nums1 = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', t1))
                nums2 = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', t2))
                
                if nums1 and nums2 and nums1 != nums2:
                    # Check if they share significant non-stopword vocabulary to ensure they talk about the same thing
                    words1 = set(re.findall(r'\b[a-z]{5,}\b', t1))
                    words2 = set(re.findall(r'\b[a-z]{5,}\b', t2))
                    overlap = len(words1 & words2)
                    
                    if overlap >= 3:
                        return f"\n\n[CONTRADICTION DETECTED]: Sources '{sources[i]}' and '{sources[j]}' appear to discuss similar topics but contain conflicting numerical data ({nums1} vs {nums2})."
                        
        return ""
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Handles:
        - Standard sentence endings (. ! ?)
        - Newline-separated items
        - Bullet points and numbered lists
        """
        # Split on sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        # Filter out empty strings and very short fragments
        return [s.strip() for s in sentences if s.strip()]
    
    def _select_diverse_sentences(
        self,
        sentences: List[dict],
        max_sentences: int = 5
    ) -> List[dict]:
        """
        Select sentences maximizing both relevance and source diversity.
        
        Uses a greedy algorithm: iteratively select the highest-scoring
        sentence from an unseen source, falling back to seen sources
        when all sources are represented.
        """
        selected = []
        used_sources = set()
        remaining = list(sentences)
        
        while len(selected) < max_sentences and remaining:
            # Try to pick from a new source first
            best = None
            for sent in remaining:
                if sent["source"] not in used_sources:
                    if best is None or sent["relevance"] > best["relevance"]:
                        best = sent
            
            # If all sources are represented, pick the best remaining
            if best is None:
                best = remaining[0]
            
            selected.append(best)
            used_sources.add(best["source"])
            remaining.remove(best)
        
        return selected


# ============================================================
# CLI Entry Point (for testing)
# ============================================================
if __name__ == "__main__":
    from router import QueryRouter, RoutingResult, QueryType
    from retriever import Retriever, RetrievalResult
    
    console.rule("[bold]Generator Test[/bold]")
    
    # Initialize components
    retriever = Retriever()
    router = QueryRouter(corpus_texts=retriever.get_all_chunk_texts())
    generator = AnswerGenerator()
    
    test_queries = [
        "What is supervised learning?",
        "Compare CNNs and RNNs",
        "What is the recipe for chocolate cake?",
    ]
    
    for query in test_queries:
        console.rule(f"\n[bold]Query: {query}[/bold]")
        
        # Retrieve
        retrieval_result = retriever.retrieve(query)
        
        # Route
        routing_result = router.route(
            query,
            retrieval_top_score=retrieval_result.top_score,
            retrieval_avg_score=retrieval_result.avg_score
        )
        console.print(f"Route: {routing_result.query_type.value} (conf={routing_result.confidence:.2f})")
        
        # Generate
        answer = generator.generate(query, routing_result, retrieval_result)
        console.print(f"Method: {answer.generation_method}")
        console.print(f"Answer: {answer.answer[:200]}...")
        console.print(f"Sources: {answer.sources}")
        console.print()
