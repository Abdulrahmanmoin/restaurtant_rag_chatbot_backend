# =============================================================================
# Dockerfile for FastAPI Backend on Hugging Face Spaces
# =============================================================================
# This Dockerfile is optimized for:
# - Hugging Face Spaces (runs on port 7860)
# - CPU-only free tier
# - Production-ready with multi-stage building
# - Models loaded once at startup, not per request
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create and use virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Lean production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Create non-root user for security (HF Spaces requirement)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create cache directories with proper permissions
RUN chown -R user:user /app

# Copy application files
COPY --chown=user:user main.py .
COPY --chown=user:user chatbot.py .
COPY --chown=user:user vector_store.py .
COPY --chown=user:user systemPrompt.txt .
COPY --chown=user:user KB.json .
COPY --chown=user:user RAG.md .

# Copy .env file if it exists (for environment variables)
# Note: For production, use HF Spaces Secrets instead
COPY --chown=user:user .env* ./

# Switch to non-root user
USER user

# Expose the port required by Hugging Face Spaces
EXPOSE 7860

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# -----------------------------------------------------------------------------
# Run the FastAPI application with Uvicorn
# -----------------------------------------------------------------------------
# - host 0.0.0.0: Listen on all interfaces (required for Docker)
# - port 7860: Hugging Face Spaces requirement
# - workers 1: Single worker for CPU-only free tier (avoids OOM)
# - timeout-keep-alive 120: Longer timeout for model loading
# -----------------------------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "120"]