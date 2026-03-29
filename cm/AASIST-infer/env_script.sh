#!/bin/bash
set -e

FAIRSEQ_PATH=${FAIRSEQ_PATH:-/home/brant/Project/MLentry/envs/custom_packages/fairseq}

uv sync
uv pip install --python .venv/bin/python -e "$FAIRSEQ_PATH" --no-deps
