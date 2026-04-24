import time
import logging
from app.config import settings
from app.llm import _get_gemini_client
from app.models import PriorityPrediction, SearchResult
from app.prompts.grounded_answer import GROUNDED_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = _get_gemini_client()

    def predict_priority(self, query: str, brand: str, sources: list[SearchResult]):
        """
        Predicts priority using RAG context and Structured Outputs.
        """
        # 1. Format context from SearchResult Pydantic objects
        context_str = "\n--(--\n".join([
            f"ID: {s.tweet_id} | Brand: {s.target_brand} | "
            f"Priority: {s.priority} | Content: {s.text}"
            for s in sources
        ])

        # 2. Build the User Prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            brand=brand,
            query=query,
            context=context_str
        )

        try:
            # 3. Call Gemini with Structured Output
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                config={
                    'system_instruction': GROUNDED_SYSTEM_PROMPT,
                    'response_mime_type': 'application/json',
                    'response_schema': PriorityPrediction, # Tell Gemini to return this object!
                    'temperature': 0.1
                },
                contents=user_prompt
            )
            
            # 4. Use .parsed for a native Pydantic object (No manual JSON parsing!)
            prediction: PriorityPrediction = response.parsed
            return {
                "priority": prediction.priority,
                "answer": prediction.reasoning
            }
        except Exception as e:
            logger.error(f"LLM Prediction Error: {e}")
            return {"priority": 1, "answer": "Fallback: LLM failed to generate structured response."}

    def get_comparative_predictions(self, query: str, sources: list[SearchResult]):
        """
        Requirement 3: Compare RAG vs. Non-RAG quality for evaluation.
        """
        context_str = "\n".join([f"- {s.text}" for s in sources])
        
        # RAG Version
        rag_ans = self.client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=f"Using this context: {context_str}\n\nQuestion: {query}"
        )
        
        # Non-RAG Version
        non_rag_ans = self.client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=f"Answer this support ticket based on general knowledge: {query}"
        )
        
        return {
            "rag_answer": rag_ans.text,
            "non_rag_answer": non_rag_ans.text
        }
    
    # Add this method to the LLMService class in backend/services/llm_service.py
    def predict_zero_shot(self, query: str):
        """Step 5: LLM zero-shot priority prediction (No RAG context)."""
        prompt = f"Is this support ticket High Priority (2), Medium Priority (1), or Low Priority (0)? Ticket: {query}"
        
        start_time = time.time()
        response = self.client.models.generate_content(
            model=settings.LLM_MODEL,
            config={
                'system_instruction': "You are a support classifier. Decide the priority based ONLY on the text provided.",
                'response_mime_type': 'application/json',
                'response_schema': PriorityPrediction,
                'temperature': 0.1
            },
            contents=prompt
        )
        latency = (time.time() - start_time) * 1000
        prediction: PriorityPrediction = response.parsed
        
        return {
            "priority": prediction.priority,
            "reasoning": prediction.reasoning,
            "latency_ms": round(latency, 2),
            "cost_usd": 0.00001 # Approx cost for 1.5-flash
        }

llm_service = LLMService()