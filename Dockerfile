# ─────────────────────────────────────────────────────────────────────────────
# MedAI – Dockerfile (Final Fix for Name Collision)
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
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt Flask-Migrate==4.0.7

# Copy application source
COPY --chown=medai:medai . .

# Create runtime directories
RUN mkdir -p instance app/uploads app/ai/weights \
    && chown -R medai:medai instance app/uploads app/ai/weights

USER medai

# Environment
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

EXPOSE 10000

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/ai/health')" || exit 1

# FIXED: Use wsgi:app to avoid collision with app/ folder
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2", "--timeout", "120", "wsgi:app"]