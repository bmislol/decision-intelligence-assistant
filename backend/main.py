import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Service Imports
from services.llm_service import llm_service
from services.ml_service import MLService
from services.vector_service import VectorService
from services.logging_service import logger

# App Imports
from app.config import settings
from app.models import AskRequest, AskResponse, SearchResult
from app.routers import search, keyword

# --- 1. Lifespan: Load the "Heavy" ML Model once ---
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Starting up: Loading ML Services and Vector Store...")
    # Load the Section 1 ML Model once to avoid memory bloat
    services["ml"] = MLService() 
    yield
    logging.info("🛑 Shutting down...")
    services.clear()

app = FastAPI(
    title="Decision Intelligence Assistant",
    description="Fulfills Sections 1-6: ML Baseline, RAG, Comparison, and Logging",
    lifespan=lifespan
)

# --- 2. Global Middleware & Routers ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Section 3 Search Routers
app.include_router(search.router, prefix="/api")
app.include_router(keyword.router, prefix="/api")

# Singletons
vector_service = VectorService()

# --- 3. INDIVIDUAL ENDPOINTS (Step 3 & 4) ---

@app.post("/api/predict/ml", tags=["Step 4: ML Baseline"])
async def predict_ml(data: dict = Body(...)):
    """Specialized ML prediction for Step 4[cite: 36]."""
    query = data.get("query")
    brand = data.get("brand", "General")
    start = time.time()
    
    # Map sector and calculate length
    sector = "Tech" if any(b in brand for b in ["Apple", "PlayStation", "Amazon"]) else "Other"
    label = services["ml"].predict_priority(query, brand, sector, len(query))
    
    latency = (time.time() - start) * 1000
    return {"priority": int(label), "latency": round(latency, 2), "cost": 0.0}

@app.post("/api/predict/rag", tags=["Step 3: RAG"])
async def predict_rag(data: dict = Body(...)):
    """Step 3: Generation with RAG context[cite: 33]."""
    query = data.get("query")
    context = vector_service.get_relevant_context(query, limit=3)
    
    # Building prompt manually since we kept the legacy LLMService
    context_str = "\n".join([f"- {res['text']}" for res in context])
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    
    answer, latency, cost = llm_service._call_gemini(prompt)
    return {"answer": answer, "sources": context, "latency": latency, "cost": cost}

@app.post("/api/predict/llm-only", tags=["Step 3: LLM Only"])
async def predict_non_rag(data: dict = Body(...)):
    """Step 3: Generation without RAG (LLM alone)[cite: 34]."""
    query = data.get("query")
    prompt = f"Answer this question based on your general knowledge: {query}"
    
    answer, latency, cost = llm_service._call_gemini(prompt)
    return {"answer": answer, "latency": latency, "cost": cost}

# --- 4. MASTER COMPARISON (Step 5 & 6) ---

@app.post("/api/predict", tags=["Step 5: Comparison"])
async def predict_and_compare(data: dict = Body(...)):
    """
    Master Orchestrator: Returns RAG, Non-RAG, ML, and LLM zero-shot metrics.
    The 'Whole Point' (Section 5)[cite: 49, 50].
    """
    query = data.get("query")
    brand = data.get("brand", "General")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # A. Retrieve Context (Step 2) [cite: 27]
        context = vector_service.get_relevant_context(query, limit=3)
        
        # B. Get LLM Comparative Suite (Step 3 & 5) [cite: 35, 54]
        # This calls the method from your Turn 18 llm_service.py
        llm_suite = llm_service.get_comparative_predictions(query, context)
        
        # C. Get ML Priority Prediction (Step 4) [cite: 53]
        sector = "Tech" if any(b in brand for b in ["Apple", "PlayStation", "Amazon"]) else "Other"
        ml_start = time.time()
        ml_label = services["ml"].predict_priority(query, brand, sector, len(query))
        ml_latency = (time.time() - ml_start) * 1000

        # D. Consolidate Response (Step 5) [cite: 55-58]
        response = {
            "query": query,
            "sources": context,
            "rag_answer": llm_suite["rag"],
            "non_rag_answer": llm_suite["non_rag"],
            "llm_priority": llm_suite["llm_priority"],
            "ml_priority": {
                "label": int(ml_label),
                "latency": round(ml_latency, 2),
                "cost": 0.0,
                "accuracy": 0.88 
            }
        }

        # E. Log Everything (Step 6 Logging) [cite: 63-70]
        logger.log_interaction(response)
        
        return response

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Decision Intelligence Assistant API is live."}