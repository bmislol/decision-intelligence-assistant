from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import ChatRequest, ChatResponse, SourceCase, PredictorResult
from services.ml_service import MLService
from services.vector_service import VectorService
from services.llm_service import LLMService

# --- Lifespan: Load everything once ---
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Loading services...")
    services["ml"] = MLService()
    services["vector"] = VectorService()
    services["llm"] = LLMService()
    yield
    print("Shutting down...")
    services.clear()

app = FastAPI(title="Acme Corp Decision Intelligence", lifespan=lifespan)

# --- Endpoint 1: ML Only (The "Reflex") ---
@app.post("/predict/ml", tags=["Debug"])
async def predict_ml(request: ChatRequest):
    return services["ml"].predict_priority(request.query, request.brand)

# --- Endpoint 2: Vector Only (The "Memory") ---
@app.post("/predict/vector", tags=["Debug"])
async def predict_vector(request: ChatRequest):
    points = services["vector"].get_relevant_context(request.query)
    return [
        SourceCase(text=p.payload['text'], priority=p.payload['priority'], score=p.score) 
        for p in points
    ]

# --- Endpoint 3: LLM Only (The "Brain") ---
@app.post("/predict/llm", tags=["Debug"])
async def predict_llm(request: ChatRequest):
    # For independent LLM testing, we still fetch context first
    context = services["vector"].get_relevant_context(request.query)
    return services["llm"].predict_priority(request.query, request.brand, context)

# --- Endpoint 4: The Orchestrator (The "Product") ---
@app.post("/predict/all", response_model=ChatResponse, tags=["Production"])
async def predict_all(request: ChatRequest):
    """The master endpoint for your React dashboard."""
    # 1. Fast ML Prediction
    ml_data = services["ml"].predict_priority(request.query, request.brand)
    
    # 2. RAG Retrieval
    context_points = services["vector"].get_relevant_context(request.query)
    sources = [
        SourceCase(text=p.payload['text'], priority=p.payload['priority'], score=p.score) 
        for p in context_points
    ]
    
    # 3. LLM Reasoning
    llm_data = services["llm"].predict_priority(request.query, request.brand, context_points)
    
    return ChatResponse(
        query=request.query,
        brand=request.brand,
        ml_result=PredictorResult(**ml_data),
        llm_result=PredictorResult(**llm_data),
        sources=sources
    )