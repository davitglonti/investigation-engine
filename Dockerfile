# Multi-stage lightweight production Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Generate 50,000 corpus and Chroma database
RUN python -c "from src.ingestion.dataset_loader import DatasetLoader; DatasetLoader.load_or_generate_corpus(50000, use_huggingface=False)"

EXPOSE 8501
EXPOSE 8000

# Default to Streamlit UI
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
