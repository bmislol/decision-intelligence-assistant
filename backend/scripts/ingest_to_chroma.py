# ============================================================
# Ingestion Script — The "Great Migration"
# ============================================================
# This script loads your 780k rows, embeds them on your GPU,
# and stores them in ChromaDB. 
# ============================================================

import sys
from pathlib import Path
from tqdm import tqdm

# Add the backend directory to sys.path so we can import our app modules
#
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag.loader import load_processed_tickets
from app.rag.chunker import chunk_tickets
from app.rag.embedder import embedder
from app.rag.store import ticket_store
from app.config import settings

def run_ingestion(batch_size: int = 5000):
    print(f"📂 Loading engineered tickets from {settings.PROCESSED_DATA_PATH}...")
    
    # 1. Load the data using our custom loader
    raw_tickets = load_processed_tickets(settings.PROCESSED_DATA_PATH)
    
    # 2. Convert them into chunks (one chunk per tweet)
    chunks = chunk_tickets(raw_tickets)
    total_chunks = len(chunks)
    print(f"✅ Prepared {total_chunks} tickets for ingestion.")

    # 3. Process in batches to optimize GPU and RAM usage
    print(f"🚀 Starting GPU-powered ingestion (Batch size: {batch_size})...")
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Ingesting to ChromaDB"):
        batch = chunks[i : i + batch_size]
        
        batch_texts = [c["text"] for c in batch]
        batch_ids = [c["source"] for c in batch]
        batch_metadatas = [c["metadata"] for c in batch]
        
        # GPU Embedding
        batch_embeddings = embedder.embed_batch(batch_texts)
        
        # Store in ChromaDB
        ticket_store.add_tickets(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )

    print(f"\n✨ Ingestion Complete! {ticket_store.count()} tickets are now searchable.")

if __name__ == "__main__":
    # To test quickly, you can use a smaller slice like raw_tickets[:1000]
    run_ingestion()