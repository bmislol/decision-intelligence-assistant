# ============================================================
# Chunker — splits documents into manageable text chunks
# ============================================================
# Adjusted for Twitter data: Each tweet is treated as a 
# self-contained chunk. For longer documents, we use the 
# instructor's sentence-aware splitting.
# ============================================================

import re
from typing import List, Dict, Any

def chunk_tickets(tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Treats each pre-processed ticket as a single, perfect chunk.
    This preserves the integrity of the tweet as a 'complete thought'.
    """
    chunks = []
    for i, ticket in enumerate(tickets):
        chunks.append({
            "text": ticket["text"],
            "source": ticket["source"],
            "chunk_index": 0,  # Each ticket is its own index 0
            "metadata": ticket["metadata"]
        })
    return chunks

def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    The instructor's 'Fixed-size with overlap' strategy.
    Used for standard long documents (PDFs, Markdown files).
    """
    text = text.strip()

    if len(text) <= chunk_size:
        return [{"text": text, "source": source, "chunk_index": 0}]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Sentence-aware splitting: scan backward for a boundary 
        # so we don't cut mid-sentence.
        if end < len(text):
            for boundary in (". ", ".\n", "? ", "!\n", "\n\n"):
                last_break = text.rfind(boundary, start, end)
                if last_break > start:
                    end = last_break + len(boundary)
                    break

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        if end >= len(text):
            start = end
        else:
            # Move forward but keep the overlap
            start = max(end - overlap, start + 1)

    return chunks