from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import importlib.util
import os
import uvicorn

from router import QueryRouter
from retriever import Retriever
from generator import AnswerGenerator
from utils import logger

app = FastAPI(title="Agentic RAG System API")

# Allow CORS for local web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components globally
logger.info("Initializing RAG backend components...")
retriever = Retriever()
router = QueryRouter(corpus_texts=retriever.get_all_chunk_texts())
generator = AnswerGenerator()
logger.info("RAG components fully initialized and ready.")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    query_type: str
    confidence: float
    sources: list
    generation_method: str
    reasoning: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Agentic RAG API is running"}

# Serve frontend HTML
@app.get("/")
def serve_frontend_root():
    return FileResponse("frontend/index.html")

# Mount static folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.post("/api/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    try:
        # 1. Retrieve Context
        retrieval_result = retriever.retrieve(req.query)
        
        # 2. Route Query based on rules and semantic embedding
        routing_result = router.route(
            req.query,
            retrieval_top_score=retrieval_result.top_score,
            retrieval_avg_score=retrieval_result.avg_score
        )
        
        # 3. Generate Answer (Gemini or Extractive)
        answer = generator.generate(req.query, routing_result, retrieval_result)
        
        return {
            "answer": answer.answer,
            "query_type": answer.query_type.value,
            "confidence": answer.confidence,
            "sources": answer.sources,
            "generation_method": answer.generation_method,
            "reasoning": answer.reasoning
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing the request.")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
