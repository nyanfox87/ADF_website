#!/usr/bin/env python3
"""
FastAPI service for AASIST deepfake audio detection.
Accepts audio files and returns confidence scores and labels.
"""

import os
import sys
import uuid
import shutil
import gc
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchaudio
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, "code")
sys.path.append(code_dir)

# Custom fairseq path — override via FAIRSEQ_PATH env var if needed
fairseq_path = os.environ.get(
    "FAIRSEQ_PATH",
    "/home/brant/Project/MLentry/envs/custom_packages/fairseq"
)
if os.path.exists(fairseq_path):
    sys.path.insert(0, fairseq_path)

from model import Model
from data_utils_SSL import pad


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    release_model()


app = FastAPI(title="AASIST ITW Deepfake Detection API", lifespan=lifespan)

UPLOAD_DIR = os.path.join(current_dir, "temp", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Configuration ---
MODEL_PATH = os.path.join(current_dir, "models", "best.pth")
SSL_CKPT   = os.path.join(current_dir, "pretrained", "fairseq_xlsr2_300m.pt")
SAMPLE_RATE = 16000
CUT_OFF     = 64600
THRESHOLD   = 0.638546   # ITW threshold for bonafide logit

# Global model state
model  = None
device = None


class Args:
    def __init__(self):
        self.ssl_ckpt = SSL_CKPT


def load_model(checkpoint_path: str = MODEL_PATH):
    """Load the AASIST model from checkpoint."""
    global model, device
    if model is not None:
        return

    print(f"Loading AASIST model from: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = Args()
    model = Model(args, device).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")


def release_model():
    """Release the model and clear VRAM."""
    global model, device
    if model is None:
        return

    print("Releasing AASIST model...")
    del model
    model = None
    device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM released.")


def preprocess_audio(audio_path: str) -> torch.Tensor:
    """Load, resample, pad audio into a model-ready tensor."""
    X, fs = torchaudio.load(audio_path)

    if X.shape[0] > 1:
        X = torch.mean(X, dim=0, keepdim=True)

    if fs != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=SAMPLE_RATE)
        X = resampler(X)

    X = X.squeeze().numpy()
    X_pad = pad(X, CUT_OFF)
    return torch.tensor(X_pad, dtype=torch.float32).unsqueeze(0).to(device)


def infer_audio(audio_path: str) -> Dict[str, Any]:
    """Run inference and return label + probability dict."""
    if model is None:
        raise RuntimeError("Model not loaded")

    input_tensor = preprocess_audio(audio_path)

    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=-1)

        bonafide_score = logits[0, 1].item()
        spoof_score    = logits[0, 0].item()

        label = "BonaFide" if bonafide_score > THRESHOLD else "Spoof"

        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()

    return {
        "label": label,
        "confidence": confidence * 100,
        "probabilities": {
            "bonafide": probs[0, 1].item(),
            "spoof":    probs[0, 0].item(),
        },
        "logits": {
            "bonafide": bonafide_score,
            "spoof":    spoof_score,
        },
        "threshold": THRESHOLD * 100,
    }


@app.get("/")
async def root():
    return {
        "name": "AASIST Deepfake Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/status": "GET  - Model status and VRAM usage",
            "/toggle": "POST - Load (mode=1) or release (mode=0) the model",
            "/detect": "POST - Detect deepfake audio (upload audio file)",
        },
    }


@app.get("/status")
async def get_status():
    status = {
        "model_loaded": model is not None,
        "device": str(device) if device else "None",
        "threshold": THRESHOLD * 100,
        "vram": {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0},
    }
    if torch.cuda.is_available():
        status["device"] = torch.cuda.get_device_name(0)
        status["vram"]["allocated_mb"]     = torch.cuda.memory_allocated() / 1024**2
        status["vram"]["reserved_mb"]      = torch.cuda.memory_reserved()  / 1024**2
        status["vram"]["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    return status


@app.post("/toggle")
async def toggle_model(mode: int = Form(...)):
    """mode=1 to load, mode=0 to release."""
    if mode == 0:
        release_model()
        return {"message": "Model released and VRAM cleared"}
    elif mode == 1:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"message": "Model loaded"}
    raise HTTPException(status_code=400, detail="Invalid mode. Use 0 or 1.")


@app.post("/detect")
async def detect_deepfake(audio: UploadFile = File(...)):
    """Detect whether an audio file is deepfake or bonafide."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /toggle with mode=1 first.")

    task_id    = str(uuid.uuid4())
    audio_path = os.path.join(UPLOAD_DIR, f"{task_id}_{audio.filename}")

    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        result = infer_audio(audio_path)
        print(f"Result: {result['label']} (confidence: {result['confidence']:.2f}%)")
        return JSONResponse(content=result)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"Warning: could not remove temp file {audio_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
