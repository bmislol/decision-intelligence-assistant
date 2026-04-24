import json
import time
from google import genai
from app.config import settings

class LLMService:
    def __init__(self):
        # Initializing the latest Google GenAI SDK (v1.73+)
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_id = settings.LLM_MODEL

    def predict_priority(self, query: str, brand: str, context_results: list):
        """Handles both raw dicts from the API and SearchResult objects from scripts."""
        start_time = time.time()
        
        context_str = ""
        for i, res in enumerate(context_results):
            # Universal access for dicts (API) or objects (evaluate.py)
            if isinstance(res, dict):
                text = res.get('text', 'No text')
                priority = res.get('metadata', {}).get('priority', 'N/A')
            else:
                text = getattr(res, 'text', 'No text')
                priority = getattr(res, 'priority', 'N/A')
            context_str += f"\n--- Case {i+1} ---\nText: {text}\nPriority: {priority}\n"

        prompt = f"Context:\n{context_str}\n\nUser: {query}\n\nReturn JSON: {{\"priority\": 0, \"answer\": \"reason\"}}"

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        
        try:
            result = json.loads(response.text.strip().strip('```json').strip('```'))
            return {"priority": int(result["priority"]), "answer": result["answer"]}
        except:
            return {"priority": 1 if "high" in response.text.lower() else 0, "answer": response.text[:100]}

llm_service = LLMService()