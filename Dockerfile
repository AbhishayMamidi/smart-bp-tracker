# Production Dockerfile for SMART-BP Blood Pressure Tracker
# Multi-stage optimized for Render Web Service deployment

FROM python:3.11-slim

# System configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    HOST=0.0.0.0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

# Install essential runtime libraries for PyTorch OpenMP, GLib, and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, static web assets, test images, and official model weights
COPY api_server.py .
COPY smart_bp_inference.py .
COPY database.py .
COPY cli_infer.py .
COPY static/ ./static/
COPY test_images/ ./test_images/
COPY 21269694/ ./21269694/

# Prepare persistent data mount point for Render Persistent Disks
RUN mkdir -p /var/data && chmod 777 /var/data

# Expose Render standard port
EXPOSE 10000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/api/health || exit 1

# Launch the existing FastAPI server
CMD ["python", "api_server.py"]
