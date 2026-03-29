#!/usr/bin/env python3
"""
FastAPI service for MFA-Conformer deepfake audio detection.
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
from torchinfo import summary

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.utils.validation import load_model_from_checkpoint
from infer_onlyone import preprocess_audio


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    release_model()


app = FastAPI(title="MFA-Conformer Deepfake Detection API", lifespan=lifespan)

UPLOAD_DIR = "temp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints/MLAADv8_LayerwiseConformerClassifier_20260123_231333"
DEFAULT_CHECKPOINT = f"{CHECKPOINT_DIR}/best_checkpoint.pt"
RESULT_FILE = f"{CHECKPOINT_DIR}/eval_InTheWild_results.txt"
MODEL_CLASS = "LayerwiseConformerClassifier"
SAMPLE_RATE = 16000
TARGET_LENGTH_SEC = 5.0

# Global model state
model = None
device = None

threshold = float(
    [line for line in open(RESULT_FILE) if 'Threshold:' in line][0].split(':')[1].strip()
)


def load_model(checkpoint_path: str = DEFAULT_CHECKPOINT):
    """Load the MFA-Conformer model from checkpoint."""
    global model, device
    if model is not None:
        return

    print(f"Loading model from: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_class_name=MODEL_CLASS,
        device=device,
    )
    model.eval()
    print("Model loaded successfully.")


def release_model():
    """Release the model and clear VRAM."""
    global model, device
    if model is None:
        return

    print("Releasing model...")
    del model
    model = None
    device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM released.")


def infer_audio(audio_path: str) -> Dict[str, Any]:
    """Run inference on an audio file and return a result dict."""
    if model is None:
        raise RuntimeError("Model not loaded")

    waveform, length = preprocess_audio(
        audio_path=audio_path,
        sample_rate=SAMPLE_RATE,
        target_length_sec=TARGET_LENGTH_SEC,
        repeat_pad=True,
    )

    waveform = waveform.to(device)
    lengths = torch.tensor([length]).to(device)

    with torch.no_grad():
        logits = model(waveform, lengths)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()

    label = "BonaFide" if pred_class == 0 else "Spoof"
    bonafide_prob = probs[0, 0].item()
    spoof_prob = probs[0, 1].item()

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "bonafide": bonafide_prob,
            "spoof": spoof_prob,
        },
        "logits": {
            "bonafide": logits[0, 0].item(),
            "spoof": logits[0, 1].item(),
        },
        "threshold": threshold,
    }


@app.get("/")
async def root():
    return {
        "name": "MFA-Conformer Deepfake Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/status": "GET  - Model status and VRAM usage",
            "/toggle": "POST - Load (mode=1) or release (mode=0) the model",
            "/detect": "POST - Detect deepfake audio (upload audio file)",
        },
    }


@app.get("/status")
async def get_status():
    """Return model status and VRAM statistics."""
    status = {
        "model_loaded": model is not None,
        "vram": {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0},
        "device": "cpu",
        "threshold": threshold,
    }

    if torch.cuda.is_available():
        status["device"] = torch.cuda.get_device_name(0)
        status["vram"]["allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        status["vram"]["reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        status["vram"]["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    if model is not None:
        try:
            model_stats = summary(model, verbose=0)
            status["model_info"] = {
                "total_params": model_stats.total_params,
                "trainable_params": model_stats.trainable_params,
                "model_size_mb": model_stats.total_param_bytes / 1024**2,
            }
        except Exception as e:
            status["model_info_error"] = str(e)

    return status


@app.post("/toggle")
async def toggle_model(mode: int = Form(...)):
    """Toggle model loading. mode=1 to load, mode=0 to release."""
    if mode == 0:
        release_model()
        return {"message": "Model released and VRAM cleared"}
    elif mode == 1:
        load_model()
        return {"message": "Model loaded"}
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 0 to release or 1 to load.")


@app.post("/detect")
async def detect_deepfake(audio: UploadFile = File(...)):
    """
    Detect whether an audio file is deepfake or bonafide.

    Returns JSON with label, confidence, probabilities, logits, and threshold.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /toggle with mode=1 first.")

    task_id = str(uuid.uuid4())
    audio_path = None

    try:
        audio_path = os.path.join(UPLOAD_DIR, f"{task_id}_{audio.filename}")
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        print(f"Processing: {audio.filename}")

        result = infer_audio(audio_path)
        result["confidence"] = result["confidence"] * 100   # percentage
        result["threshold"] = result["threshold"] * 100     # percentage

        print(f"Result: {result['label']} (confidence: {result['confidence']:.2f}%)")

        return JSONResponse(content=result)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                print(f"Warning: could not remove temp file {audio_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
