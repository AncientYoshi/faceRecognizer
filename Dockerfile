FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/service \
    FACE_DATABASE_PATH=/service/data/faces.db \
    INSIGHTFACE_MODEL_ROOT=/home/service/.insightface

WORKDIR /service

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 service \
    && mkdir -p /service/data /home/service/.insightface \
    && chown -R service:service /service /home/service
USER service

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# Keep one worker: each worker will hold its own InsightFace model in memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

STOPSIGNAL SIGTERM
