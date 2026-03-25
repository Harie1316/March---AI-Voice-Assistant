# ──────────────────────────────────────────────
#  AI Voice Assistant — Production Dockerfile
#  Multi-stage build for optimised image size
# ──────────────────────────────────────────────

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    libffi-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Production image
FROM python:3.11-slim

LABEL maintainer="AI Voice Assistant"
LABEL version="1.0.0"
LABEL description="Production-grade voice assistant with speech domain adaptation"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /tmp/voice_assistant_uploads /app/logs \
    && chown -R appuser:appuser /app /tmp/voice_assistant_uploads

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_DEBUG=false \
    PORT=5000 \
    WHISPER_MODEL=base \
    LOG_LEVEL=INFO \
    WORKER_THREADS=4 \
    MAX_CONCURRENT=50

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Expose port
EXPOSE ${PORT}

# Switch to non-root user
USER appuser

# Run with Gunicorn for production
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "run:create_app()"]
