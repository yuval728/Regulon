# Multi-stage Dockerfile for Regulon serving
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Add local packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY serve/ ./serve/
COPY eval/ ./eval/
COPY quantize/ ./quantize/

# Model artifacts will be mounted at runtime or copied in
# COPY ./merged_16bit ./model/

# Expose port
EXPOSE 8000

# Environment variables
ENV ENGINE_TYPE=vllm
ENV MODEL_PATH=/app/model
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "-m", "serve.api"]