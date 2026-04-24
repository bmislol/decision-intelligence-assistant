import json
import time
from google import genai
from app.config import settings

class LLMService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_id = settings.LLM_MODEL
        # 2026 pricing for gemini-1.5-flash: ~$0.075 / 1M tokens
        self.cost_per_token = 0.000000075 

    def _call_gemini(self, prompt: str):
        start_time = time.time()
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        latency = (time.time() - start_time) * 1000
        # Estimate tokens (approx 4 chars per token)
        tokens = len(prompt + response.text) / 4
        cost = tokens * self.cost_per_token
        
        return response.text, round(latency, 2), round(cost, 6)

    def get_comparative_predictions(self, query: str, context_results: list):
        """
        Generates RAG Answer, Non-RAG Answer, and Zero-Shot Priority.
        """
        # 1. Prepare RAG Context (Fix: Use dictionary keys)
        context_str = "\n".join([
            f"- Past Case: {res.get('text', 'N/A')} | Priority: {res.get('metadata', {}).get('priority', 'N/A')}" 
            for res in context_results
        ])

        # 2. Generate RAG Answer
        rag_prompt = f"Use this context to answer: {context_str}\n\nQuestion: {query}"
        rag_ans, rag_lat, rag_cost = self._call_gemini(rag_prompt)

        # 3. Generate Non-RAG Answer
        non_rag_prompt = f"Answer this question based on your general knowledge: {query}"
        nr_ans, nr_lat, nr_cost = self._call_gemini(non_rag_prompt)

        # 4. Zero-Shot Priority
        priority_prompt = f"Is this support ticket High Priority (1) or Low Priority (0)? Ticket: {query}. Return ONLY the number."
        p_raw, p_lat, p_cost = self._call_gemini(priority_prompt)
        
        try:
            llm_priority = int(''.join(filter(str.isdigit, p_raw)))
        except:
            llm_priority = 1 if "high" in p_raw.lower() else 0

        return {
            "rag": {"answer": rag_ans, "latency": rag_lat, "cost": rag_cost},
            "non_rag": {"answer": nr_ans, "latency": nr_lat, "cost": nr_cost},
            "llm_priority": {"label": llm_priority, "latency": p_lat, "cost": p_cost}
        }

llm_service = LLMService()