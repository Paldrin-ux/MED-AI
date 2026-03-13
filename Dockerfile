# ─────────────────────────────────────────────────────────────────────────────
# MedAI – Dockerfile
# Base: Python 3.11 slim (smaller image, faster CI)
# GPU: swap base image to nvidia/cuda:12.1.0-runtime-ubuntu22.04 + install torch+cu121
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# OS dependencies for OpenCV, pydicom pixel data, and libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    gdcm \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 medai

WORKDIR /app

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=medai:medai . .

# Create runtime directories
RUN mkdir -p instance app/uploads app/ai/weights \
    && chown -R medai:medai instance app/uploads app/ai/weights

USER medai

# Environment
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

EXPOSE 5000

# Health-check endpoint (requires /ai/health to return 200)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/ai/health')" || exit 1

# Production: gunicorn with 2 workers (CPU), 4 threads
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
