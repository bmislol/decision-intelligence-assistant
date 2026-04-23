from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Decision Intelligence Assistant"
    
    # Paths (Anchored to your project root)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CHROMA_DB_PATH: str = str(BASE_DIR / "data" / "chroma_db")
    ML_MODEL_PATH: str = str(BASE_DIR / "backend" / "models" / "priority_model.joblib")
    PROCESSED_DATA_PATH: str = str(BASE_DIR / "data" / "processed_tickets.joblib")
    
    # Gemini Config
    GEMINI_API_KEY: str = ""
    # 'text-embedding-004' is the current gold standard for Gemini embeddings
    EMBEDDING_MODEL: str = "models/text-embedding-004" 
    # 'gemini-1.5-flash' is great for speed/cost, 'pro' is better for logic
    LLM_MODEL: str = "gemini-1.5-pro" 

    class Config:
        # This tells Pydantic to look for your .env file
        env_file = ".env"
        extra = "ignore"

settings = Settings()