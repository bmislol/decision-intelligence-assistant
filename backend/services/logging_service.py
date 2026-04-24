import json
import os
from datetime import datetime
from pathlib import Path

class LoggingService:
    def __init__(self):
        self.log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "query_history.jsonl"

    def log_interaction(self, payload: dict):
        """Saves interaction metadata to a JSONL file for Step 6 compliance."""
        payload["timestamp"] = datetime.utcnow().isoformat()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(payload) + "\n")

logger = LoggingService()