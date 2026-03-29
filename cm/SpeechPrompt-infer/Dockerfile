FROM nvidia/cuda:12.4.0-base-ubuntu20.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app:/app/code \
    FAIRSEQ_PATH=/app/fairseq

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Clone fairseq and apply compatibility patches
RUN git clone --depth 1 https://github.com/facebookresearch/fairseq.git fairseq && \
    sed -i 's/def build_encoder_layer(self, args):/def build_encoder_layer(self, args, **kwargs):/' \
        fairseq/fairseq/models/wav2vec/wav2vec2.py && \
    sed -i 's/def extract_features(self, x, padding_mask=None, tgt_layer=None):/def extract_features(self, x, padding_mask=None, tgt_layer=None, **kwargs):/' \
        fairseq/fairseq/models/wav2vec/wav2vec2.py && \
    sed -i "s/if label is not \"\" else/if label != \"\" else/" \
        fairseq/fairseq/tasks/multires_hubert_pretraining.py

# Build environment via env_script.sh (same script used outside Docker)
COPY env_script.sh ./
RUN chmod +x env_script.sh && ./env_script.sh

# Copy source code and config
COPY code/ ./code/
COPY config.yaml ./
COPY speechprompt_fastapi.py ./

# Copy data (dict.txt + verbalizer.json — small files, baked into image)
COPY data/ ./data/

RUN mkdir -p temp/uploads

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "speechprompt_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
