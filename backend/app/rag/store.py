# ============================================================
# Vector Store — ChromaDB wrapper for storing and searching
# ============================================================
# Stores your 780k tweets as vectors for semantic search.
# Uses Cosine similarity as per instructor's standard.
# ============================================================

import chromadb
from app.config import settings
from typing import List, Dict, Any

COLLECTION_NAME = "twitter_support_tickets"

class TicketStore:
    def __init__(self):
        # Create a persistent client that saves to your data/ folder
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        
        # Get or create the collection with Cosine Distance
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} #
        )

    def add_tickets(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Store tickets in ChromaDB. 
        Uses 'upsert' to prevent duplicates if you run the script twice.
        """
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most similar tickets.
        Returns the text, metadata, and the distance score.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        items = []
        # Parse the results into a clean list of dicts for our API
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        return items

    def count(self) -> int:
        """Returns the number of tickets currently indexed."""
        return self.collection.count()

    def clear(self):
        """Deletes everything for a fresh start."""
        self.client.delete_collection(name=COLLECTION_NAME)

# Singleton instance for the app
ticket_store = TicketStore()