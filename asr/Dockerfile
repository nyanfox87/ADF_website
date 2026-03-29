FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi uvicorn[standard] pydantic aiofiles

# Install Python dependencies
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git

RUN pip install python-multipart

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app