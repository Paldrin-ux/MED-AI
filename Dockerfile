# ─────────────────────────────────────────────────────────────────────────────
# MedAI – Dockerfile (Final Final Fix: Permissions & Dependencies)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# OS dependencies for OpenCV, pydicom, and libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgdcm-tools \
    python3-gdcm \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 medai

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# Ensure all critical libraries are present
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt Flask-Migrate==4.0.7 requests==2.31.0

# Copy application source
COPY --chown=medai:medai . .

# Create directories and FORCE permissions so the app can save uploaded images
RUN mkdir -p instance app/uploads app/ai/weights \
    && chmod -R 777 /app/instance /app/app/uploads /app/app/ai/weights \
    && chown -R medai:medai /app

USER medai

# Environment
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

EXPOSE 10000

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/ai/health')" || exit 1

# FIXED: Points to wsgi.py to avoid the 'app' folder collision
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2", "--timeout", "120", "wsgi:app"]
