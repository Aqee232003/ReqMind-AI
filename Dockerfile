# ── Dockerfile for ReqMind AI ──────────────────────────────────
# Google Cloud Run expects a container that listens on PORT env var (default 8080)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by faiss and sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Cloud Run sets PORT env var — uvicorn must listen on it
ENV PORT=8080

# Start the FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
