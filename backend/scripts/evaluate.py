import sys
from pathlib import Path
import pandas as pd

import time

# Add backend to path so we can import our modules
# This assumes the script is in decision-intelligence-assistant/backend/scripts/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.models import SearchResult 
from services.ml_service import MLService
from services.llm_service import llm_service
from app.rag.store import ticket_store
from app.rag.embedder import embedder

def run_evaluation():
    print("🧪 Starting Evaluation: ML vs. RAG vs. Non-RAG")
    
    # 1. Initialize Services
    ml_service = MLService()
    
    # 2. Define Test Cases (Categorized for sector logic)
    test_cases = [
        {"brand": "AppleSupport", "text": "my screen is flickering and I see green lines", "sector": "Tech"},
        {"brand": "AmazonHelp", "text": "I ordered a laptop and the box arrived empty!!", "sector": "Retail"},
        {"brand": "Uber_Support", "text": "The driver was very polite and the car was clean.", "sector": "Service"},
        {"brand": "AskPlayStation", "text": "I can't log in to my account and I have a tournament in 10 minutes", "sector": "Tech"},
    ]

    results = []

    for case in test_cases:
        query = case["text"]
        brand = case["brand"]
        sector = case["sector"]
        print(f"\n📝 Testing: {query[:50]}...")

        # --- Step 1: ML Prediction (Section 1 Baseline) ---
        ml_priority = ml_service.predict_priority(query, brand, sector, len(query))

        # --- Step 2: RAG Retrieval (Section 2) ---
        query_vector = embedder.embed_text(query)
        raw_sources = ticket_store.search(query_vector, top_k=3)
        
        # Convert to SearchResult objects for the service
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
        
        # --- Step 3: LLM Predictions (Section 3 Comparison) ---
        # A. RAG-Powered Priority & Reasoning
        rag_result = llm_service.predict_priority(query, brand, sources)
        
        # B. Comparative Generation (RAG vs. Non-RAG Answer Quality)
        comparative = llm_service.get_comparative_predictions(query, sources)

        results.append({
            "Query": query,
            "ML_Priority": int(ml_priority),
            "LLM_RAG_Priority": rag_result["priority"],
            "LLM_RAG_Reasoning": rag_result["answer"],
            "RAG_Answer": comparative["rag_answer"],
            "Non_RAG_Answer": comparative["non_rag_answer"]
        })

        print("😴 Waiting to respect Free Tier rate limits...")
        time.sleep(20)

    # 3. Save and Display Comparison
    df = pd.DataFrame(results)
    
    print("\n📊 Priority Comparison (ML vs LLM):")
    print(df[["Query", "ML_Priority", "LLM_RAG_Priority"]])
    
    print("\n💡 Sample RAG Reasoning:")
    print(df["LLM_RAG_Reasoning"].iloc[0])

    # Save to models/ for documentation
    output_path = Path(__file__).parent.parent / "models" / "evaluation_report.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Detailed Comparison Report saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()