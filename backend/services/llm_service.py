from app.llm import call_llm
from app.prompts.grounded_answer import GROUNDED_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

class LLMService:
    def predict_priority(self, query: str, brand: str, context_results: list) -> dict:
        """
        Takes the RAG results and asks Gemini to make a final decision.
        Fulfills Section 4: The LLM Predictor.
        """
        # 1. Format the historical context for the prompt
        context_str = ""
        for i, res in enumerate(context_results):
            # We use the text and metadata we stored in ChromaDB
            context_str += f"\n--- Historical Case {i+1} ---\n{res.text}\n"

        # 2. Build the user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=query,
            brand=brand,
            context=context_str
        )

        # 3. Call Gemini
        raw_response = call_llm(GROUNDED_SYSTEM_PROMPT, user_prompt)

        # 4. Return the reasoning and a predicted priority 
        # (We'll parse the priority from the text for now)
        predicted_priority = 1
        if "priority: 2" in raw_response.lower() or "high priority" in raw_response.lower():
            predicted_priority = 2
        elif "priority: 0" in raw_response.lower() or "low priority" in raw_response.lower():
            predicted_priority = 0

        return {
            "answer": raw_response,
            "priority": predicted_priority
        }

llm_service = LLMService()