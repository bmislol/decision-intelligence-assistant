import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Project Imports
from app.config import settings
from app.models import AskRequest, AskResponse, SearchResult
from app.routers import search, keyword
from services.ml_service import MLService
from app.rag.store import ticket_store
from app.rag.embedder import embedder

services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Starting up: Loading ML Services and Vector Store...")
    # Initialize the ML service
    services["ml"] = MLService() 
    yield
    logging.info("🛑 Shutting down...")
    services.clear()

app = FastAPI(
    title="Decision Intelligence Assistant",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(search.router)
app.include_router(keyword.router)

@app.post("/predict/ml", tags=["Section 1"])
async def predict_ml(request: AskRequest):
    try:
        # Static sector map for demo purposes
        sector = "Tech" if "Apple" in request.brand or "PlayStation" in request.brand else "Other"
        prediction = services["ml"].predict_priority(
            query=request.question, 
            brand=request.brand,
            sector=sector,
            text_len=len(request.question)
        )
        return {"priority": int(prediction), "model": "LogisticRegression"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/all", response_model=AskResponse, tags=["Section 4"])
async def predict_all(request: AskRequest):
    # ML Prediction
    sector = "Tech" if "Apple" in request.brand or "PlayStation" in request.brand else "Other"
    ml_priority = services["ml"].predict_priority(request.question, request.brand, sector, len(request.question))
    
    # RAG Retrieval
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
    
    return AskResponse(
        answer="Retrieval complete. Pending LLM Reasoning integration.",
        ml_priority=int(ml_priority),
        rag_priority=None, 
        sources=sources,
        latency_ms={"total": 0.1}
    )