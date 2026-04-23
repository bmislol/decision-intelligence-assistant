# ============================================================
# Loader — reads raw or processed data into plain text
# ============================================================
# Adjusted for the Twitter Decision Intelligence Project.
# This module converts your engineered joblib data and 
# standard files into plain text strings for the chunker.
# ============================================================

import joblib
import csv
import json
import io
from pathlib import Path
from typing import List, Dict, Any

# Purpose: Load your engineered tickets from the joblib file
def load_processed_tickets(path: str) -> List[Dict[str, Any]]:
    """
    Loads the joblib file and converts rows into a format 
    compatible with the RAG pipeline.
    """
    df = joblib.load(path)
    
    # We return a list of dicts where 'text' is the content 
    # and other fields are metadata, mirroring the instructor's 
    # chunk structure.
    tickets = []
    for _, row in df.iterrows():
        # We create a 'grounding string' that includes context 
        # for the LLM.
        content = (
            f"Ticket ID: {row['tweet_id']}\n"
            f"Brand: {row['target_brand']} ({row['brand_sector']})\n"
            f"Priority: {row['priority']}\n"
            f"Content: {row['clean_text']}"
        )
        tickets.append({
            "text": content,
            "source": f"tweet_{row['tweet_id']}",
            "metadata": {
                "tweet_id": str(row['tweet_id']),
                "priority": int(row['priority']),
                "brand": row['target_brand'],
                "sector": row['brand_sector']
            }
        })
    return tickets

# Purpose: Keep the instructor's general-purpose loader for extra docs
# (Like company policies or help manuals you might add later)
def load_file(path: str) -> str:
    """Read a standard file and return content as plain text."""
    filepath = Path(path)
    ext = filepath.suffix.lower()

    if ext in (".md", ".txt"):
        return filepath.read_text(encoding="utf-8")
    elif ext == ".csv":
        return _load_csv(filepath)
    elif ext == ".json":
        return _load_json(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _load_csv(filepath: Path) -> str:
    """Convert CSV into 'Column: Value' blocks."""
    text = filepath.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = [f"\n".join([f"{col}: {val}" for col, val in row.items()]) for row in reader]
    return "\n\n".join(rows)

def _load_json(filepath: Path) -> str:
    """Flatten JSON into readable text."""
    text = filepath.read_text(encoding="utf-8")
    data = json.loads(text)
    return json.dumps(data, indent=2)