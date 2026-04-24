import os
import joblib
from fastapi import APIRouter, Query
from app.models import SearchResponse, SearchResult
from app.config import settings

router = APIRouter(prefix="/keyword-search", tags=["Search"])

# --- FIX: Join the directory from settings with the actual filename ---
# This ensures we load the file, not the folder
data_file_path = os.path.join(settings.PROCESSED_DATA_PATH, "processed_tickets.joblib")

# Check if the path is still a directory (just in case) and fix it
if os.path.isdir(data_file_path):
    data_file_path = os.path.join(data_file_path, "processed_tickets.joblib")

df = joblib.load(data_file_path)

@router.get("", response_model=SearchResponse)
def search_keywords(
    q: str = Query(description="The exact keyword to search for."),
    top_k: int = Query(default=5)
):
    """Standard keyword search — searching without the RAG system."""
    matches = df[df['clean_text'].str.contains(q, case=False, na=False)].head(top_k)
    
    results = []
    for _, row in matches.iterrows():
        results.append(SearchResult(
            tweet_id=str(row['tweet_id']),
            text=row['clean_text'],
            priority=int(row['priority']),
            target_brand=row['target_brand'],
            brand_sector=row['brand_sector'],
            distance=0.0
        ))
    
    return SearchResponse(query=q, results=results, total_results=len(results))