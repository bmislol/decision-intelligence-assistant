from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# --- 1. THE SEARCH LAYER (Needed for search.py) ---
class SearchRequest(BaseModel):
    """The incoming search contract"""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class SearchResult(BaseModel):
    """A single 'hit' from our 780k tickets"""
    tweet_id: str
    text: str
    priority: int
    target_brand: str
    brand_sector: str
    distance: float  # Semantic similarity score

class SearchResponse(BaseModel):
    """The final package sent to the UI"""
    query: str
    results: List[SearchResult]
    total_results: int

# --- 2. THE MESSAGE & CONVERSATION LAYER ---
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = "New Conversation"
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)

# --- 3. THE AI ASSISTANT LAYER ---
class AskRequest(BaseModel):
    question: str
    use_rag: bool = True

class AskResponse(BaseModel):
    answer: str
    ml_priority: int           # Our Logistic Regression result
    rag_priority: Optional[int] # What the LLM thinks
    sources: List[SearchResult]
    latency_ms: Dict[str, float]

# --- 4. THE LOGGING & DEBUG LAYER ---
class SystemLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    level: str = Field(default="INFO")  # INFO, WARNING, ERROR
    source: str                         # e.g., "ML_MODEL", "CHROMA_DB"
    message: str
    metadata: Optional[Dict[str, Any]] = None