# CPU app image. The GPU generator runs separately (vLLM) and is reached over
# LLM_BASE_URL — one env var, no docker-compose (see plan §10).
FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/models PYTHONUNBUFFERED=1

# App needs the light stack + retrieval/serving libs, not vllm.
COPY requirements-ci.txt .
RUN pip install -r requirements-ci.txt \
    && pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" \
       "sentence-transformers>=2.7" "faiss-cpu>=1.7.4" \
       "transformers>=4.40" "openai>=1.30" "trafilatura>=1.8"

COPY artifacts/gate.joblib /app/artifacts/gate.joblib
COPY config.yaml /app/config.yaml
COPY trustrag/ /app/trustrag/
COPY eval/ /app/eval/
COPY service/ /app/service/

EXPOSE 8080
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
