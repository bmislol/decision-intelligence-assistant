from pydantic import BaseModel, Field
from typing import List, Optional

# --- Input Model ---
class ChatRequest(BaseModel):
    query: str = Field(..., example="My order hasn't arrived yet, I need a refund.")
    brand: str = Field(..., example="AppleSupport")

# --- Supporting Models ---
class SourceCase(BaseModel):
    """Represents a past case retrieved via RAG."""
    text: str
    priority: int
    score: float # The semantic similarity score from Qdrant

class PredictorResult(BaseModel):
    """A unified result model for both ML and LLM predictors."""
    priority: int
    latency_ms: float
    cost_usd: float
    answer: Optional[str] = None # ML won't have an answer, but LLM will

# --- Final Output Model ---
class ChatResponse(BaseModel):
    """The complete response bundle for the comparison dashboard."""
    query: str
    brand: str
    ml_result: PredictorResult
    llm_result: PredictorResult
    sources: List[SourceCase]