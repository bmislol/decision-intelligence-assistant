import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class VectorService:
    def __init__(self):
        # Load embedding model once
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Connect to local Qdrant
        self.qdrant = QdrantClient(path=os.getenv("QDRANT_PATH", "../data/qdrant_db"))
        self.collection = os.getenv("COLLECTION_NAME", "support_cases")

    def get_relevant_context(self, query: str, limit: int = 2):
        """Finds semantic matches in the vector store."""
        vector = self.embed_model.encode(query).tolist()
        
        # Using the new query_points API we verified earlier
        search_response = self.qdrant.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit
        )
        
        return search_response.points