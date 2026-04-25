# Decision Intelligence Assistant

An AI-driven support ticket classification system that leverages local Machine Learning and Retrieval-Augmented Generation (RAG) to prioritize and answer customer inquiries. This project demonstrates a "Decision Intelligence" approach by comparing the accuracy, latency, and cost of different AI strategies: Local ML, RAG-LLM, and Zero-shot LLM.

## 🚀 Quick Start (Docker)

The fastest way to get the entire stack running is using Docker Compose. This ensures all services (FastAPI, React, ChromaDB) are correctly networked.

1. **Clone the repository** and navigate to the root directory.
2. **Create a `.env` file** in the `backend/` directory (see the Environment Variables section below).
3. **Run the stack**:
   ```bash
   docker compose up --build
   ```
### Access the Application:

- **Frontend**: http://localhost:5173

- **Backend API**: http://localhost:8000

- **API Docs**: http://localhost:8000/docs

### 🛠️ Local Setup & Development
If you prefer to run the services manually for development, follow this specific order of execution to prepare the data and models.

1. Prerequisites
Python 3.12+ (Using uv for package management is highly recommended)

Node.js 20+

NVIDIA GPU (Optional, but the system is optimized for CUDA/RTX 3060)

2. Backend Setup
```bash
cd backend
uv sync
```
3. Data & Model Pipeline (Execution Order)
You must run these scripts in order to build the intelligence layer:

Data Cleaning: Preprocesses the raw Twitter dataset (780k+ tickets).

```bash
uv run python scripts/clean_data.py
```

ML Training: Trains the local Gradient Boosting baseline model.

```bash
uv run python scripts/train_baseline.py
```

Vector Ingestion: Embeds the tickets and stores them in ChromaDB.

```bash
uv run python scripts/ingest_to_chroma.py
```

4. Start the Services
Backend:

```bash
uv run uvicorn main:app --reload
```

Frontend:

```bash
cd ../frontend
npm install
npm run dev
```

### 🔑 Environment Variables
Create a .env file inside the backend/ folder.

Code snippet
# Google Gemini API Key
GOOGLE_API_KEY=your_api_key_here

# LLM Configuration
LLM_MODEL=gemini-2.5-flash

# Database Configuration
CHROMA_DB_PATH=data/chroma_db
### 🏗️ Project Architecture
Frontend: React 19 + Vite 8 + Tailwind CSS v4. Features an Everforest Dark Hard theme and a real-time comparison dashboard.

Backend: FastAPI. Handles RAG orchestration, local ML predictions, and background administration tasks.

RAG Engine:

Embedder: all-mpnet-base-v2 running locally on CUDA.

Vector Store: ChromaDB in persistent mode.

ML Service: Scikit-Learn based priority classifier for zero-cost baseline predictions.

📊 Deployment & Orchestration
This project is fully containerized using Docker Compose.

Shared Network: Services communicate via a dedicated bridge network.

Persistent Volumes: ChromaDB data and system logs are persisted in ./backend/data and ./backend/logs to ensure data survives container restarts.