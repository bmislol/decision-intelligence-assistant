import os
import time
import joblib
import pandas as pd

class MLService:
    def __init__(self):
        # Determine the path to the model
        # Using a relative path that works whether you run from backend/ or root
        model_path = os.getenv("ML_MODEL_PATH", "models/priority_model.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Did you run Phase 2?")
            
        # Load the full pipeline (Preprocessor + Classifier)
        self.pipeline = joblib.load(model_path)
        print(f"ML Service: Loaded model from {model_path}")

    def predict_priority(self, text: str, brand: str):
        """
        Runs a prediction using the 'Champion' Linear SVC model.
        Returns priority and latency in milliseconds.
        """
        # Create a DataFrame because the Scikit-Learn Pipeline expects 
        # the same features/columns it saw during training.
        input_df = pd.DataFrame([{
            "clean_text": text,
            "author_id": brand
        }])
        
        start_time = time.time()
        
        # Run the prediction
        prediction = self.pipeline.predict(input_df)[0]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "priority": int(prediction),
            "latency_ms": round(latency_ms, 4), # 4 decimals to show how fast it is!
            "cost_usd": 0.0 # Traditional ML inference is effectively free
        }