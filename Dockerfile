
# ==========================================
# AI EMOTION DETECTION - DOCKERFILE
# ==========================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/

# Copy trained model checkpoint
COPY models/emotion-distilbert/ ./models/emotion-distilbert/

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
