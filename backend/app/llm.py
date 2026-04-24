import logging
from functools import lru_cache
from google import genai
from app.config import settings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_gemini_client():
    """Cached client factory following instructor's pattern."""
    return genai.Client(api_key=settings.GEMINI_API_KEY)

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Sends a grounded prompt to Gemini.
    Uses the newer google-genai SDK as seen in the repo.
    """
    client = _get_gemini_client()
    
    try:
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            config={
                'system_instruction': system_prompt,
                'temperature': 0.1, # Keep it deterministic for priority prediction
            },
            contents=user_prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        raise