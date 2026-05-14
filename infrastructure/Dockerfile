FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY backend/ backend/
COPY ml/ ml/

ARG MODEL_DIR=outputs/onnx/quantized
COPY ${MODEL_DIR} /app/model/

ENV MODEL_DIR=/app/model \
    MAX_LENGTH=512 \
    PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/docs')"

CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
