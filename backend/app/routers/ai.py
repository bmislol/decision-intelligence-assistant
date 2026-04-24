import time
from fastapi import APIRouter, Request, HTTPException
from app.models import (
    AskRequest, RAGResponse, LLMOnlyResponse, 
    MLResponse, ZeroShotResponse, MasterResponse,
    SearchResult
)
from services.llm_service import llm_service
from app.rag.store import ticket_store
from app.rag.embedder import embedder

from services.logging_service import logger

router = APIRouter(prefix="/predict", tags=["Step 5: Comparison"])

@router.post("/rag", response_model=RAGResponse)
async def predict_rag(request: AskRequest):
    """Output 1: The RAG answer (LLM + retrieved context)."""
    start_time = time.time()
    query_vector = embedder.embed_text(request.question)
    raw_hits = ticket_store.search(query_vector, top_k=3)
    
    sources = [
        SearchResult(
            tweet_id=s["metadata"]["tweet_id"],
            text=s["text"],
            priority=s["metadata"]["priority"],
            target_brand=s["metadata"]["brand"],
            brand_sector=s["metadata"]["sector"],
            distance=s["distance"]
        ) for s in raw_hits
    ]
    
    result = llm_service.predict_priority(request.question, "General", sources)
    res = RAGResponse(
        answer=result["answer"],
        sources=sources,
        latency_ms=round((time.time() - start_time) * 1000, 2),
        cost_usd=0.00002
    )

    logger.log_interaction("RAG_LLM", request.question, res.model_dump())

    return res

@router.post("/llm-only", response_model=LLMOnlyResponse)
async def predict_llm_only(request: AskRequest):
    """Output 2: The non-RAG answer (LLM alone)."""
    start_time = time.time()
    comparative = llm_service.get_comparative_predictions(request.question, [])
    return LLMOnlyResponse(
        answer=comparative["non_rag_answer"],
        latency_ms=round((time.time() - start_time) * 1000, 2),
        cost_usd=0.00001
    )

@router.post("/ml", response_model=MLResponse)
async def predict_ml(request: Request, data: AskRequest):
    """Output 3: The ML priority prediction (Local Classifier)."""
    start_time = time.time()
    # Access the model from the app state
    ml_service = request.app.state.ml_service
    priority = ml_service.predict_priority(data.question, "General", "Other", len(data.question))
    
    res = MLResponse(
        priority=int(priority),
        latency_ms=round((time.time() - start_time) * 1000, 2),
        cost_usd=0.0
    )

    logger.log_interaction("ML_ONLY", data.question, res.model_dump())

    return res

@router.post("/zero-shot", response_model=ZeroShotResponse)
async def predict_zero_shot(data: AskRequest):
    """Output 4: The LLM zero-shot priority prediction (No Context)."""
    result = llm_service.predict_zero_shot(data.question)
    return ZeroShotResponse(**result)

@router.post("/all", response_model=MasterResponse)
async def predict_all(request: Request, data: AskRequest):
    """The Master Orchestrator: Calls all 4 logic paths."""
    start = time.time()
    
    # We can literally call our other functions internally or just use the services
    # To save your quota tomorrow, we just use the service calls directly here
    try:
        query_vector = embedder.embed_text(data.question)
        raw_hits = ticket_store.search(query_vector, top_k=3)
        sources = [SearchResult(tweet_id=s["metadata"]["tweet_id"], text=s["text"], 
                priority=s["metadata"]["priority"], target_brand=s["metadata"]["brand"],
                brand_sector=s["metadata"]["sector"], distance=s["distance"]) for s in raw_hits]

        rag_res = llm_service.predict_priority(data.question, "General", sources)
        comp_res = llm_service.get_comparative_predictions(data.question, [])
        ml_res = request.app.state.ml_service.predict_priority(data.question, "General", "Other", len(data.question))
        zs_res = llm_service.predict_zero_shot(data.question)

        response = MasterResponse(
            query=data.question,
            rag=RAGResponse(answer=rag_res["answer"], sources=sources, latency_ms=0.0, cost_usd=0.00002),
            llm_only=LLMOnlyResponse(answer=comp_res["non_rag_answer"], latency_ms=0.0, cost_usd=0.00001),
            ml=MLResponse(priority=int(ml_res), latency_ms=0.0, cost_usd=0.0),
            zero_shot=ZeroShotResponse(**zs_res),
            total_latency_ms=round((time.time() - start) * 1000, 2)
        )

        logger.log_interaction(response.model_dump())

        return response
    
    except Exception as e:
        logger.log_error("MASTER_API", str(e), {"query": data.question})
        raise HTTPException(status_code=500, detail="Internal AI Processing Error")