# Multi-Architecture Base (Supports linux/arm64 on AWS Graviton4 & linux/amd64 locally)
FROM --platform=$TARGETPLATFORM python:3.11-slim

LABEL maintainer="GuardianCV Team"
LABEL description="GuardianCV: OpenCV 5 + AWS Graviton COOL Active Perception Rehab Agent"

# Enable Arm64 NEON / SVE2 hardware acceleration hints for OpenCV / COOL runtime
ENV OPENCV_OPT_ENABLE_NEON=1
ENV OMP_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies for OpenCV headless execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]