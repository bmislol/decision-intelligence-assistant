# backend/services/logging_service.py
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

class LoggingService:
    def __init__(self):
        # Anchor to the project root logs folder
        self.log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        self.interaction_file = self.log_dir / "query_history.jsonl"
        self.system_file = self.log_dir / "system_errors.jsonl"

    def log_interaction(self, source: str, input_query: str, response: dict):
        """
        Saves a successful interaction.
        'source' tells you which endpoint was used (ML, RAG, etc.)
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "query": input_query,
                "response": response
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            self.log_error("LOGGING_SERVICE", f"Failed to log interaction: {str(e)}")

    def log_error(self, source: str, message: str, metadata: Dict = None):
        """Dedicated error logging for debugging."""
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "ERROR",
            "source": source,
            "message": message,
            "metadata": metadata or {}
        }
        with open(self.system_file, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
        logging.error(f"[{source}] {message}")

logger = LoggingService()