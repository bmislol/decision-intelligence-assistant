import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from services.ml_service import MLService
from app.routers import search, keyword, ai # Import your new router

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

# Register all routers
app.include_router(search.router, prefix="/api")
app.include_router(keyword.router, prefix="/api")
app.include_router(ai.router, prefix="/api")

@app.get("/")
def root():
    return {"status": "online", "model": settings.LLM_MODEL}