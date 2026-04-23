# ============================================================
# Router: /search — Test retrieval in isolation
# ============================================================
# Use this to verify that your 780k tickets are actually 
# findable by meaning. Lower distance = higher similarity.
# ============================================================

from fastapi import APIRouter, Query
from app.models import SearchRequest, SearchResult
from app.rag.embedder import embedder
from app.rag.store import ticket_store

router = APIRouter(prefix="/search", tags=["Search"])

@router.get("")
def search_tickets(
    q: str = Query(description="The search query text."),
    top_k: int = Query(default=5, description="Number of results to return.")
):
    """Retrieval without generation — proving the math works."""
    
    # 1. Check if we have data
    if ticket_store.count() == 0:
        return {"query": q, "results": [], "total_results": 0}

    # 2. Use your RTX 3060 to embed the query locally
    query_vector = embedder.embed_text(q)
    
    # 3. Search ChromaDB
    raw_results = ticket_store.search(query_vector, top_k=top_k)

    # 4. Format for the API
    results = [
        SearchResult(
            tweet_id=r["metadata"]["tweet_id"],
            text=r["text"],
            clean_text=r["text"], # Our processed text is the doc
            priority=r["metadata"]["priority"],
            target_brand=r["metadata"]["brand"],
            brand_sector=r["metadata"]["sector"],
            distance=r["distance"]
        )
        for r in raw_results
    ]

    return {"query": q, "results": results, "total_results": len(results)}