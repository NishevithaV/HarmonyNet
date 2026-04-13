FROM python:3.12-slim

WORKDIR /app

# MuseScore dependencies (for PDF rendering)
RUN apt-get update && apt-get install -y \
    musescore3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
# basic-pitch pulls in TensorFlow on Linux — use requirements-docker.txt which
# excludes it, then install basic-pitch without its deps + onnxruntime manually.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
RUN pip install --no-cache-dir basic-pitch --no-deps

COPY api/requirements.txt api_requirements.txt
RUN pip install --no-cache-dir -r api_requirements.txt

COPY . .
