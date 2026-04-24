import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

class LoggingService:
    def __init__(self):
        # Anchor to the project root logs folder
        self.log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Use consistent names for the log files
        self.interaction_file = self.log_dir / "query_history.jsonl"
        self.system_file = self.log_dir / "system_errors.jsonl"

    def log_interaction(self, source: str, input_query: str, response_data: Dict[str, Any]):
        """Saves a successful interaction for Step 6 compliance."""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "query": input_query,
                "response": response_data
            }
            with open(self.interaction_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.log_error("LOGGING_SERVICE", f"Failed to log interaction: {str(e)}")

    def log_error(self, source: str, message: str, metadata: Optional[Dict] = None):
        """Dedicated error logging for Step 6 compliance."""
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