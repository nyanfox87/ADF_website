#!/usr/bin/env python3
"""
Single audio inference script for MFA-Conformer deepfake detection.
Usage: python infer_onlyone.py --audio <path> --checkpoint <path>
"""

import os
import sys
import argparse
import torch
import torchaudio
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.validation import load_model_from_checkpoint


def preprocess_audio(
    audio_path: str,
    sample_rate: int = 16000,
    target_length_sec: float = 5.0,
    repeat_pad: bool = True,
) -> tuple:
    """
    Load and preprocess a single audio file.

    Returns:
        waveform: Tensor [1, num_samples]
        length: int
    """
    waveform, orig_sr = torchaudio.load(audio_path)

    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)

    target_length = int(sample_rate * target_length_sec)
    current_length = waveform.shape[0]

    if current_length > target_length:
        waveform = waveform[:target_length]
    elif current_length < target_length:
        if repeat_pad:
            num_repeats = (target_length // current_length) + 1
            waveform = waveform.repeat(num_repeats)[:target_length]
        else:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

    length = waveform.shape[0]
    waveform = waveform.unsqueeze(0)  # [1, num_samples]

    return waveform, length


def infer_single_audio(
    audio_path: str,
    checkpoint_path: str,
    model_class: str = "LayerwiseConformerClassifier",
    sample_rate: int = 16000,
    target_length_sec: float = 5.0,
    device: str = None,
) -> dict:
    """
    Run inference on a single audio file.

    Returns dict with:
        prediction, confidence, bonafide_prob, spoof_prob,
        bonafide_logit, spoof_logit
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model_from_checkpoint(checkpoint_path, device, model_class)
    model.eval()

    print(f"Loading audio: {audio_path}")
    waveform, length = preprocess_audio(
        audio_path,
        sample_rate=sample_rate,
        target_length_sec=target_length_sec,
        repeat_pad=True,
    )

    waveform = waveform.to(device)
    length_tensor = torch.tensor([length], dtype=torch.long).to(device)

    print("Running inference...")
    with torch.no_grad():
        logits = model(input_signal=waveform, input_signal_length=length_tensor)
        probs = torch.softmax(logits, dim=1)

        bonafide_logit = logits[0, 0].item()
        spoof_logit = logits[0, 1].item()
        bonafide_prob = probs[0, 0].item()
        spoof_prob = probs[0, 1].item()

        prediction = "bonafide" if bonafide_prob > spoof_prob else "spoof"
        confidence = max(bonafide_prob, spoof_prob)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "bonafide_prob": bonafide_prob,
        "spoof_prob": spoof_prob,
        "bonafide_logit": bonafide_logit,
        "spoof_logit": spoof_logit,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inference for a single audio file with MFA-Conformer"
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--model_class",
        type=str,
        default="LayerwiseConformerClassifier",
        help="Model class name (default: LayerwiseConformerClassifier)",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--target_length_sec", type=float, default=5.0)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])

    args = parser.parse_args()

    try:
        result = infer_single_audio(
            audio_path=args.audio,
            checkpoint_path=args.checkpoint,
            model_class=args.model_class,
            sample_rate=args.sample_rate,
            target_length_sec=args.target_length_sec,
            device=args.device,
        )

        print("\n" + "=" * 60)
        print("INFERENCE RESULT")
        print("=" * 60)
        print(f"Audio file:  {args.audio}")
        print(f"Prediction:  {result['prediction'].upper()}")
        print(f"Confidence:  {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"\nDetailed probabilities:")
        print(f"  Bonafide: {result['bonafide_prob']:.4f} ({result['bonafide_prob']*100:.2f}%)")
        print(f"  Spoof:    {result['spoof_prob']:.4f} ({result['spoof_prob']*100:.2f}%)")
        print(f"\nRaw logits:")
        print(f"  Bonafide: {result['bonafide_logit']:.4f}")
        print(f"  Spoof:    {result['spoof_logit']:.4f}")
        print("=" * 60)

        sys.exit(0 if result['prediction'] == 'bonafide' else 1)

    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
