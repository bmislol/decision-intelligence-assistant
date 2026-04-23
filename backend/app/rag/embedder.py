# ============================================================
# Embedder — turns text into mathematical "meaning" vectors
# ============================================================
# Optimized for Charbel's Legion 5 RTX 3060 GPU.
# This runs 100% locally on Ubuntu with CUDA support.
# ============================================================

import torch
from sentence_transformers import SentenceTransformer
from typing import List
import time

class LocalEmbedder:
    def __init__(self):
        # 1. Choose model: 'all-mpnet-base-v2' = 768 dimensions (Instructor's standard)
        self.model_name = 'all-mpnet-base-v2'
        
        # 2. Check for GPU (RTX 3060)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Initializing Local Embedder on device: {self.device.upper()}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
    def embed_text(self, text: str) -> List[float]:
        """Convert a single string to a vector."""
        # result is a numpy array, we convert to list for ChromaDB compatibility
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of strings to vectors.
        Uses GPU batch processing for massive speedups on 780k rows.
        """
        start_time = time.time()
        
        # We use a larger batch_size since you have 6GB VRAM
        embeddings = self.model.encode(
            texts, 
            batch_size=128, 
            show_progress_bar=False, 
            convert_to_tensor=False
        )
        
        elapsed = time.time() - start_time
        # print(f"⚡ Embedded {len(texts)} chunks in {elapsed:.2f}s")
        
        return embeddings.tolist()

# Instantiate the singleton for the app to use
embedder = LocalEmbedder()