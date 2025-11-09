# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    GOOGLE_GENAI_USE_VERTEXAI=true \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies required for matplotlib and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (located in same directory as Dockerfile in ERAgent folder)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with increased timeout and parallel downloads
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout 300 -r /app/requirements.txt

# Copy application code (same directory level as Dockerfile)
COPY ER_Vitals_Monitoring_Assistant/ ./ER_Vitals_Monitoring_Assistant/
COPY fastapi_app.py /app/fastapi_app.py
COPY static/ /app/static/

# Copy service account key (if needed - uncomment and add the file)
# COPY spiritual-clock-471207-i1-bd9f5d8c36f0.json /app/

# Set environment variable for service account (if using key file)
# ENV GOOGLE_APPLICATION_CREDENTIALS=/app/spiritual-clock-471207-i1-bd9f5d8c36f0.json

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8080

# Health check - updated for FastAPI endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=5)" || exit 1

# Run FastAPI with uvicorn
# Cloud Run provides PORT environment variable, uvicorn will use it
CMD ["sh", "-c", "uvicorn fastapi_app:app --host 0.0.0.0 --port ${PORT:-8080}"]
