# backend/services/vector_service.py
from app.rag.store import ticket_store
from app.rag.embedder import embedder

class VectorService:
    def get_relevant_context(self, query: str, limit: int = 2):
        # 1. Embed the query using your local GPU embedder
        query_vector = embedder.embed_text(query)
        
        # 2. Search ChromaDB (ticket_store)
        results = ticket_store.search(query_vector, top_k=limit)
        return results # Returns list of dicts with 'text', 'metadata', 'distance'