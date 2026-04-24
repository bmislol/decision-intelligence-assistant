import os
import joblib
import pandas as pd
from app.config import settings

class MLService:
    def __init__(self):
        # Use absolute path from central config
        model_path = settings.ML_MODEL_PATH
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
            
        # Load the full pipeline once
        self.pipeline = joblib.load(model_path)
        print(f"ML Service: Loaded model from {model_path}")

    def predict_priority(self, query: str, brand: str, sector: str, text_len: int):
        """
        Runs a prediction using the 4 features your pipeline expects.
        """
        input_df = pd.DataFrame([{
            "clean_text": query,
            "author_id": brand,
            "brand_sector": sector,
            "text_len": text_len
        }])
        
        prediction = self.pipeline.predict(input_df)[0]
        return int(prediction)

ml_service = MLService()