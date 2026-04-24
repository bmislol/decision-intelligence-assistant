# backend/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "Decision Intelligence Assistant"
    
    # Path logic
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
    ROOT_DIR: Path = BACKEND_DIR.parent
    
    # Map .env keys to these variables
    ml_model_path_raw: str = Field(default="models/priority_model.joblib", alias="ML_MODEL_PATH")
    chroma_db_path_raw: str = Field(default="data/chroma_db", alias="CHROMA_DB_PATH")
    
    GEMINI_API_KEY: str = ""
    # FIX: Use the stable 2026 production ID to avoid 404s
    LLM_MODEL: str = "gemini-1.5-flash" 

    @property
    def ML_MODEL_PATH(self) -> str:
        path = Path(self.ml_model_path_raw)
        return str(self.BACKEND_DIR / path) if not path.is_absolute() else str(path)

    @property
    def CHROMA_DB_PATH(self) -> str:
        # data/ is in the project root, not inside backend/
        return str(self.ROOT_DIR / "data" / "chroma_db")
    
    @property
    def PROCESSED_DATA_PATH(self) -> str:
        """Points to the directory where processed_tickets.joblib lives."""
        return str(self.ROOT_DIR / "data")

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        extra="ignore",
        populate_by_name=True
    )

settings = Settings()