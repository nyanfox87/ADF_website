FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    python3.10 \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-install-project || uv sync --no-install-project

# Copy source code
COPY src/ ./src/
COPY mfa_fastapi.py infer_onlyone.py ./

# Create temp upload directory
RUN mkdir -p temp/uploads

EXPOSE 8000
