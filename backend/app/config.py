import os
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
    # Updated: Use 'gemini-1.5-flash' as it's more stable for testing
    LLM_MODEL: str = "gemini-1.5-flash" 
    EMBEDDING_MODEL: str = "text-embedding-004"

    class Config:
        # Use the absolute path so it works from any directory
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        extra = "ignore"

settings = Settings()