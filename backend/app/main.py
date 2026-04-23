from fastapi import FastAPI
from app.routers import search

app = FastAPI(title="Decision Intelligence Assistant")

# Include the search router we just built
app.include_router(search.router)

@app.get("/")
def read_root():
    return {"message": "API is live. Go to /docs for Swagger UI"}