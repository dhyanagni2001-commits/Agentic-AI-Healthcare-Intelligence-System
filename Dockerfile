# HealthIQ API (FastAPI + FAISS + sentence-transformers), CPU-only.
# Data CSVs and the FAISS index are mounted at /app/data at runtime, not baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app
# CPU torch wheel first so sentence-transformers doesn't pull CUDA wheels (~2.5 GB) on amd64.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model so pods start without Hugging Face network access.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

COPY backend/ backend/
COPY benchmarks/ benchmarks/
RUN useradd -u 10001 -m app && mkdir -p /app/data && chown -R app /app/data /opt/hf
USER 10001

EXPOSE 8000
CMD ["uvicorn", "backend.main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
