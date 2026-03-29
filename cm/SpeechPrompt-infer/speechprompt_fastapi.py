#!/usr/bin/env python3
"""
FastAPI service for SpeechPrompt v2 deepfake audio detection.
Pipeline: audio file → HuBERT features → KMeans units → verbalize → GSLM prompt → label
"""

import gc
import json
import os
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ── sys.path setup ─────────────────────────────────────────────────────────────
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
code_dir = current_dir / "code"
sys.path.insert(0, str(code_dir))

fairseq_path = os.environ.get(
    "FAIRSEQ_PATH",
    "/home/brant/Project/MLentry/envs/custom_packages/fairseq",
)
if os.path.exists(fairseq_path):
    sys.path.insert(0, fairseq_path)

# Register custom fairseq components (model/task/loss) before any fairseq call
import fairseq_usr.model  # noqa: registers GSLM_prompt architecture
import fairseq_usr.task   # noqa: registers prompt_language_modeling task
import fairseq_usr.loss   # noqa: registers cross_entropy_prompt criterion

from fairseq import checkpoint_utils, options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from speech2unit.hubert_feature_reader import HubertFeatureReader

# ── config ─────────────────────────────────────────────────────────────────────
import yaml

with open(current_dir / "config.yaml") as f:
    _cfg = yaml.safe_load(f)
PROMPT_PARAM_FILTER: list = _cfg["prompt_param_filter"]  # ["sep", "prompt", "verbalizer"]

HUBERT_CKPT     = current_dir / "pretrained" / "hubert_base_ls960.pt"
KMEANS_PATH     = current_dir / "pretrained" / "km.bin"
BASE_MODEL_PATH = current_dir / "checkpoints" / "base_prompt_model.pt"
PROMPT_PATH     = current_dir / "checkpoints" / "checkpoint_best.pt"
DATA_BIN_DIR    = current_dir / "data"          # must contain dict.txt
VERBALIZER_PATH = current_dir / "data" / "verbalizer.json"

PROMPT_LENGTH = 5
HUBERT_LAYER  = 6
SAMPLE_RATE   = 16000
MAX_GEN_LEN   = 150

UPLOAD_DIR = current_dir / "temp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── global state ────────────────────────────────────────────────────────────────
hubert_reader: HubertFeatureReader = None
kmeans_model  = None
sp_model      = None
sp_task       = None
verbalizer: Dict[str, str] = None   # unit_token → label  (reverse of verbalizer.json)
device        = None


# ── lifecycle ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    release_model()


app = FastAPI(title="SpeechPrompt v2 Deepfake Detection API", lifespan=lifespan)


# ── model management ───────────────────────────────────────────────────────────
def _build_fairseq_args():
    """Build a minimal fairseq args namespace for task setup + model loading."""
    parser = options.get_interactive_generation_parser()
    arg_list = [
        str(DATA_BIN_DIR),
        f"--path={BASE_MODEL_PATH}",
        "--task=prompt_language_modeling",
        "--sampling",
        "--sampling-topk=1",
        "--seed=1",
        "--max-len-a=0",
        f"--max-len-b={MAX_GEN_LEN}",
        "--batch-size=1",
        "--bf16",
        "--skip-invalid-size-inputs-valid-test",
        f"--model-overrides={{'prompt_length': {PROMPT_LENGTH}}}",
    ]
    args = options.parse_args_and_arch(parser, arg_list)
    try:
        args = convert_namespace_to_omegaconf(args)
    except Exception:
        pass
    return args


def load_model():
    global hubert_reader, kmeans_model, sp_model, sp_task, verbalizer, device

    if sp_model is not None:
        return

    for path, label in [
        (BASE_MODEL_PATH, "base model"),
        (PROMPT_PATH,     "prompt checkpoint"),
        (HUBERT_CKPT,     "HuBERT checkpoint"),
        (KMEANS_PATH,     "KMeans model"),
        (VERBALIZER_PATH, "verbalizer"),
        (DATA_BIN_DIR / "dict.txt", "dictionary"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    print("Loading HuBERT feature reader …")
    hubert_reader = HubertFeatureReader(
        checkpoint_path=str(HUBERT_CKPT),
        layer=HUBERT_LAYER,
    )

    print("Loading KMeans model …")
    kmeans_model = joblib.load(str(KMEANS_PATH))
    kmeans_model.verbose = False

    print("Loading verbalizer …")
    with open(VERBALIZER_PATH) as f:
        fwd = json.load(f)          # {"spoof": "0", "bonafide": "1"}
    verbalizer = {v: k for k, v in fwd.items()}   # {"0": "spoof", "1": "bonafide"}

    print("Setting up fairseq task …")
    args = _build_fairseq_args()
    task_cfg = args.task if hasattr(args, "task") else args
    sp_task_local = tasks.setup_task(task_cfg)

    print(f"Loading GSLM base model from: {BASE_MODEL_PATH}")
    overrides = {"prompt_length": PROMPT_LENGTH, "data": str(DATA_BIN_DIR)}
    models, _ = checkpoint_utils.load_model_ensemble(
        [str(BASE_MODEL_PATH)],
        arg_overrides=overrides,
        task=sp_task_local,
    )

    print(f"Merging learned prompts from: {PROMPT_PATH}")
    model_dict = models[0].state_dict()
    prompt_dict = torch.load(str(PROMPT_PATH), map_location="cpu")["model"]
    for name in model_dict:
        for filter_name in PROMPT_PARAM_FILTER:
            if filter_name in name and name in prompt_dict:
                model_dict[name] = prompt_dict[name]
    models[0].load_state_dict(model_dict)

    device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_local}")

    models[0].bfloat16()
    models[0] = models[0].to(device_local)
    models[0].eval()

    sp_model  = models[0]
    sp_task   = sp_task_local
    device    = device_local
    print("SpeechPrompt model loaded successfully.")


def release_model():
    global hubert_reader, kmeans_model, sp_model, sp_task, verbalizer, device

    if sp_model is None and hubert_reader is None:
        return

    print("Releasing SpeechPrompt models …")
    del sp_model, hubert_reader, kmeans_model
    sp_model = hubert_reader = kmeans_model = sp_task = verbalizer = device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM released.")


# ── inference pipeline ─────────────────────────────────────────────────────────
def _audio_to_units(audio_path: str) -> list:
    """Audio file → list of int speech unit IDs (consecutive-same reduced)."""
    feats = hubert_reader.get_feats(audio_path)          # [T, dim] on CUDA
    feats_np = feats.cpu().numpy()
    pred = kmeans_model.predict(feats_np)                 # [T] int array

    # reduce consecutive same units
    reduced = []
    prev = -1
    for unit in pred:
        if unit != prev:
            reduced.append(int(unit))
            prev = unit
    return reduced


def infer_audio(audio_path: str) -> Dict[str, Any]:
    """Full pipeline: audio → label + confidence."""
    if sp_model is None:
        raise RuntimeError("Model not loaded")

    # 1. Audio → speech units
    units = _audio_to_units(audio_path)

    # 2. Format verbalized input: "u1 u2 u3 ... <s>"
    src_str = " ".join(str(u) for u in units) + " <s>"

    # 3. Encode with fairseq dictionary
    src_dict = sp_task.source_dictionary
    tokens = src_dict.encode_line(src_str, add_if_not_exist=False).long()
    src_tokens  = tokens.unsqueeze(0).to(device)
    src_lengths = torch.tensor([tokens.numel()]).to(device)

    sample = {
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        }
    }

    # 4. Autoregressive generation (linear verbalizer mode)
    sp_model.eval()
    prediction: list = []
    last_confidence = 1.0

    with torch.no_grad():
        for _ in range(MAX_GEN_LEN):
            output, _ = sp_model(**sample["net_input"])
            pred_logits = output[:, -1, :]                       # [1, num_classes]
            pred_probs  = torch.softmax(pred_logits.float(), dim=-1)
            pred_idx    = int(np.argmax(pred_logits.float().cpu().numpy()))

            if pred_idx == src_dict.eos():
                break

            last_confidence = pred_probs[0, pred_idx].item()

            new_tok = torch.tensor([[pred_idx]], device=device)
            src_tokens  = torch.cat((sample["net_input"]["src_tokens"], new_tok), dim=1)
            src_lengths = sample["net_input"]["src_lengths"] + 1
            sample = {"net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}}

            if pred_idx >= 4:                                    # skip special tokens
                prediction.append(str(pred_idx - 4))            # dict index → unit ID

    # 5. Decode label
    pred_unit = " ".join(prediction)
    label = verbalizer.get(pred_unit, "Unknown")

    return {
        "label": label,
        "confidence": last_confidence * 100,
        "raw_prediction": pred_unit,
    }


# ── endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "SpeechPrompt v2 Deepfake Detection API",
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
        "model_loaded": sp_model is not None,
        "device": str(device) if device else "None",
        "vram": {"allocated_mb": 0, "reserved_mb": 0},
    }
    if torch.cuda.is_available():
        status["device"] = torch.cuda.get_device_name(0)
        status["vram"]["allocated_mb"] = torch.cuda.memory_allocated() / 1024 ** 2
        status["vram"]["reserved_mb"]  = torch.cuda.memory_reserved()  / 1024 ** 2
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
    if sp_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /toggle with mode=1 first.")

    task_id    = str(uuid.uuid4())
    audio_path = str(UPLOAD_DIR / f"{task_id}_{audio.filename}")

    try:
        with open(audio_path, "wb") as buf:
            shutil.copyfileobj(audio.file, buf)

        result = infer_audio(audio_path)
        print(f"Result: {result['label']} (confidence: {result['confidence']:.2f}%)")
        return JSONResponse(content=result)

    except Exception as e:
        print(f"Error: {e}")
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
