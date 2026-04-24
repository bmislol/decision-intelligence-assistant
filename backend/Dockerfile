FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy the rest of the code
COPY . .

# Ensure logs and data directories exist
RUN mkdir -p logs data/chroma_db

EXPOSE 8000

# Run the app (host 0.0.0.0 is required for Docker)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]