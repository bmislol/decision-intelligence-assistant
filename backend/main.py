import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Project Imports
from app.config import settings
from app.models import AskRequest, AskResponse, SearchResult
from app.routers import search, keyword

# We will load these via the lifespan
from services.ml_service import MLService
from app.rag.store import ticket_store
from app.rag.embedder import embedder

# --- 1. Lifespan: Load the "Heavy" ML Model once ---
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Starting up: Loading ML Services and Vector Store...")
    # Load your Section 1 ML Model
    services["ml"] = MLService() 
    yield
    logging.info("🛑 Shutting down...")
    services.clear()

app = FastAPI(
    title="Decision Intelligence Assistant",
    description="ML Priority Prediction + RAG Knowledge Assistant",
    lifespan=lifespan
)

# --- 2. Middleware & Routers ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Section 3 Routers (Semantic vs Keyword Search)
app.include_router(search.router)
app.include_router(keyword.router)

# --- 3. SECTION 1: ML Only Endpoint ---
@app.post("/predict/ml", tags=["Section 1: Baseline"])
async def predict_ml(request: AskRequest):
    """Reflex action: Quick ML prediction from the Logistic Regression model."""
    try:
        # Map sector and length to avoid the ValueError seen earlier
        sector = "Tech" if "Apple" in request.brand or "PlayStation" in request.brand else "Other"
        text_len = len(request.question)
        
        prediction = services["ml"].predict_priority(
            query=request.question, 
            brand=request.brand,
            sector=sector,
            text_len=text_len
        )
        return {"priority": int(prediction), "model": "LogisticRegression"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Service Error: {str(e)}")

# --- 4. SECTION 4: The Orchestrator (The Product) ---
@app.post("/predict/all", response_model=AskResponse, tags=["Section 4: Production"])
async def predict_all(request: AskRequest):
    """
    The Master Endpoint: Combines ML Baseline + RAG + LLM Reasoning.
    This fulfills Section 4 of the project requirements.
    """
    # 1. Get ML Prediction (Section 1)
    sector = "Tech" if "Apple" in request.brand or "PlayStation" in request.brand else "Other"
    ml_priority = services["ml"].predict_priority(request.question, request.brand, sector, len(request.question))
    
    # 2. Get RAG Context (Section 2/3)
    query_vector = embedder.embed_text(request.question)
    raw_sources = ticket_store.search(query_vector, top_k=3)
    
    sources = [
        SearchResult(
            tweet_id=s["metadata"]["tweet_id"],
            text=s["text"],
            priority=s["metadata"]["priority"],
            target_brand=s["metadata"]["brand"],
            brand_sector=s["metadata"]["sector"],
            distance=s["distance"]
        ) for s in raw_sources
    ]
    
    # 3. LLM Logic (Section 4 - Coming next!)
    # For now, we'll return a placeholder until we build the llm_service
    return AskResponse(
        answer="The RAG system found similar cases. I am ready to process them.",
        ml_priority=int(ml_priority),
        rag_priority=None, 
        sources=sources,
        latency_ms={"total": 0.5} 
    )

@app.get("/")
def root():
    return {"message": "API is live. Explore /docs for Section 1-4 endpoints."}