# =============================================================
# agentic-rag Dockerfile
# =============================================================
# Builds a container for the Agentic RAG pipeline and Streamlit UI.
#
# Build:
#   docker build -t agentic-rag .
#
# Run interactive mode:
#   docker run --env-file .env agentic-rag
#
# Run single query:
#   docker run --env-file .env \
#     agentic-rag python src/pipeline.py --query "What is the PTO policy?"
#
# Run Streamlit UI:
#   docker run --env-file .env -p 8501:8501 agentic-rag \
#     streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0
#
# Notes:
#   - Unlike rag-pipeline, NO volume mount needed for the vector index
#     since Pinecone is cloud-managed — just pass API keys via --env-file
#   - Mount ./data only if you need to ingest new documents
#     docker run --env-file .env -v $(pwd)/data:/app/data agentic-rag
# =============================================================

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System dependencies for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY ui/ ./ui/
COPY evaluation/ ./evaluation/
COPY tests/ ./tests/

# Create data directories
RUN mkdir -p data/raw data/processed

# Expose Streamlit port
EXPOSE 8501

# Default command — interactive pipeline mode
CMD ["python", "src/pipeline.py"]
