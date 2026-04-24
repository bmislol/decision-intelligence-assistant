import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from services.ml_service import MLService
from app.routers import search, keyword, ai, admin # Import your new router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Starting up: Loading ML Service...")
    # Store it in app.state so routers can access it safely
    app.state.ml_service = MLService() 
    yield
    logging.info("🛑 Shutting down...")

app = FastAPI(title="Decision Intelligence Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import BackgroundTasks
import subprocess

@app.post("/api/admin/run-evaluate")
async def trigger_evaluate(background_tasks: BackgroundTasks):
    # Runs the script in the background so the API doesn't hang
    background_tasks.add_task(subprocess.run, ["python", "scripts/evaluate.py"])
    return {"status": "Evaluation started in background"}

# Register all routers
app.include_router(search.router, prefix="/api")
app.include_router(keyword.router, prefix="/api")
app.include_router(ai.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

@app.get("/")
def root():
    return {"status": "online", "model": settings.LLM_MODEL}