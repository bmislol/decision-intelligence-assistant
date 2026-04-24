import subprocess
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path

router = APIRouter(prefix="/admin", tags=["Admin Operations"])

# Get the path to the root scripts folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def run_script(script_name: str):
    """Helper to execute scripts in the scripts/ folder."""
    script_path = BASE_DIR / "scripts" / script_name
    try:
        # We use a simplified call; for a real production app, 
        # you'd use a task queue like Celery.
        subprocess.run(["python", str(script_path)], check=True)
    except Exception as e:
        print(f"Error running {script_name}: {e}")

@router.post("/run-evaluation")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """Fulfills Step 5 evaluation script trigger."""
    background_tasks.add_task(run_script, "evaluate.py")
    return {"message": "Evaluation started. Check /models/evaluation_report.csv later."}

@router.post("/run-ingestion")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Triggers the 40-minute GPU migration to ChromaDB."""
    background_tasks.add_task(run_script, "ingest_to_chroma.py")
    return {"message": "Ingestion started. This may take 30-40 minutes on your RTX 3060."}