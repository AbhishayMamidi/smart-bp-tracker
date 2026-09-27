# Production Dockerfile for SMART-BP Blood Pressure Tracker
# Hardened for Render Free ($0) headless Linux deployment

FROM python:3.11-slim

# System configuration & headless optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    HOST=0.0.0.0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    YOLO_AUTOINSTALL=0 \
    ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 \
    OMP_NUM_THREADS=2

# Install minimal runtime libraries:
# - libglib2.0-0 & libgomp1: required by OpenCV & PyTorch OpenMP
# - libxcb1 & libgl1: minimal C runtimes for XCB/GL compatibility without full GUI/desktop stack
# - curl: required by container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install dependencies, guarantee removal of any conflicting GUI OpenCV packages,
# and ensure pure headless OpenCV is installed cleanly
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --no-cache-dir --no-deps --force-reinstall opencv-python-headless==4.10.0.84

# Build-time verification Step 1: Verify cv2 imports cleanly in headless mode
# Prints version and full build information to confirm headless runtime
RUN python -c "import cv2; print('OpenCV Version:', cv2.__version__); print(cv2.getBuildInformation())"

# Copy application code, static web assets, test images, and official model weights
COPY api_server.py .
COPY smart_bp_inference.py .
COPY database.py .
COPY exif_utils.py .
COPY storage.py .
COPY dataset_manager.py .
COPY cli_infer.py .
COPY static/ ./static/
COPY test_images/ ./test_images/
COPY 21269694/ ./21269694/

# Build-time verification Step 2: Test importing all application modules —
# smart_bp_inference, api_server (which imports exif_utils, database, storage, etc.),
# exif_utils, storage, and dataset_manager directly. This catches any missing COPY or missing dependency
# before the image is pushed to the registry.
RUN python -c "import cv2; import exif_utils; import storage; import dataset_manager; import smart_bp_inference; import api_server; print('Build verification successful: all modules imported with zero missing shared libraries.')"

# Prepare storage directory for SQLite persistence
RUN mkdir -p /var/data && chmod 777 /var/data

# Expose Render standard port
EXPOSE 10000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/api/health || exit 1

# Launch FastAPI server
CMD ["python", "api_server.py"]
