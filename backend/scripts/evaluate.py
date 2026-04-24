import sys
from pathlib import Path

# Add backend to path so we can import our modules
# This makes 'app' and 'services' top-level packages
sys.path.append(str(Path(__file__).resolve().parent.parent))

# FIX: Removed 'backend.' prefix because 'backend' is already in sys.path
from app.models import SearchResult 
from services.ml_service import MLService
from services.llm_service import llm_service
from app.rag.store import ticket_store
from app.rag.embedder import embedder
import pandas as pd

def run_evaluation():
    print("🧪 Starting Section 5: ML vs. RAG+LLM Evaluation")
    
    # 1. Initialize Services
    ml_service = MLService()
    
    # 2. Define Test Cases (Tricky queries to test context)
    test_cases = [
        {"brand": "AppleSupport", "text": "my screen is flickering and I see green lines"},
        {"brand": "AmazonHelp", "text": "I ordered a laptop and the box arrived empty!!"},
        {"brand": "Uber_Support", "text": "The driver was very polite and the car was clean."},
        {"brand": "AskPlayStation", "text": "I can't log in to my account and I have a tournament in 10 minutes"},
    ]

    results = []

    for case in test_cases:
        query = case["text"]
        brand = case["brand"]
        print(f"\n📝 Testing: {query[:50]}...")

        # --- ML Prediction (Section 1) ---
        sector = "Tech" if any(b in brand for b in ["Apple", "PlayStation", "Amazon"]) else "Other"
        ml_priority = ml_service.predict_priority(query, brand, sector, len(query))

        # --- RAG + LLM Prediction (Section 4) ---
        query_vector = embedder.embed_text(query)
        raw_sources = ticket_store.search(query_vector, top_k=3)
        
        # Wrap raw dictionaries into SearchResult objects for the LLM
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
        
        # Call Gemini via the LLM service
        llm_result = llm_service.predict_priority(query, brand, sources)

        results.append({
            "Query": query,
            "ML_Priority": int(ml_priority),
            "LLM_Priority": llm_result["priority"],
            "LLM_Reasoning": llm_result["answer"][:100] + "..."
        })

    # 3. Save and Display Comparison
    df = pd.DataFrame(results)
    print("\n📊 Evaluation Summary:")
    print(df[["Query", "ML_Priority", "LLM_Priority"]])
    
    output_path = Path(__file__).parent.parent / "models" / "evaluation_report.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Report saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()