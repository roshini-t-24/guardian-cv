FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/guardian-cv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY utils/ utils/
COPY static/ static/
COPY server.py benchmark_cool.py validate_metrics.py ./

RUN python -c "from utils.video_generator import ensure_clips; ensure_clips('videos')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]