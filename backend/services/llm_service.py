import os
import time
import json
from google import genai # New import

from dotenv import load_dotenv
load_dotenv()  # This is the "magic" line that reads your .env file

class LLMService:
    def __init__(self):
        # The new client-based approach
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def predict_priority(self, query: str, brand: str, context_points: list):
        start_time = time.time()
        
        context_text = ""
        for p in context_points:
            context_text += f"- Past Tweet: {p.payload['text']} | Priority: {p.payload['priority']}\n"

        prompt = f"""
        System: You are a customer support intelligence assistant for {brand}.
        Predict priority (1 for High, 0 for Low) and provide a short response.
        
        Context:
        {context_text}
        
        User Message: {query}
        
        Return ONLY valid JSON: {{"priority": 0, "answer": "text"}}
        """

        # New syntax: client.models.generate_content
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        
        latency = (time.time() - start_time) * 1000
        
        # Clean and parse JSON (The response structure is slightly different now)
        raw_text = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(raw_text)
        
        return {
            "priority": result["priority"],
            "answer": result["answer"],
            "latency_ms": round(latency, 2),
            "cost_usd": 0.00001 
        }