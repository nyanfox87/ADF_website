"""
Inference utilities for MFA-Conformer deepfake audio detection.
"""
import torch
import importlib
import numpy as np
from tqdm import tqdm


def get_model_class(model_class_name: str):
    """
    Dynamically import a model class from src.models.conformer.

    Args:
        model_class_name: Name of the model class (e.g. 'LayerwiseConformerClassifier')

    Returns:
        Model class
    """
    try:
        model_module = importlib.import_module('src.models.conformer')

        if not hasattr(model_module, model_class_name):
            raise AttributeError(f"Model class '{model_class_name}' not found in conformer.py")

        model_class = getattr(model_module, model_class_name)
        print(f"✓ Imported model class: {model_class_name}")
        return model_class

    except ImportError as e:
        raise ImportError(f"Failed to import model module: {e}")
    except Exception as e:
        raise Exception(f"Error loading model class '{model_class_name}': {e}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    model_class_name: str = "LayerwiseConformerClassifier",
    model_kwargs: dict = None,
):
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device string ('cuda' or 'cpu')
        model_class_name: Name of the model class to instantiate
        model_kwargs: Optional keyword arguments to override model defaults

    Returns:
        model in eval mode
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    ModelClass = get_model_class(model_class_name)

    default_kwargs = {
        "base_model_name": "stt_en_conformer_ctc_small",
        "num_classes": 2,
        "trainable_encoder": False,
        "dropout": 0.5,
    }
    if model_kwargs:
        default_kwargs.update(model_kwargs)

    model = ModelClass(**default_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    if 'best_val_acc' in checkpoint:
        print(f"  Best val accuracy: {checkpoint['best_val_acc']:.2f}%")

    return model


def generate_scores(model, data_loader, device):
    """
    Run inference over a DataLoader and collect probabilities.

    Args:
        model: Model in eval mode
        data_loader: DataLoader yielding (waveforms, lengths[, labels])
        device: Device string

    Returns:
        all_probs: np.ndarray of bonafide probabilities
        all_labels: np.ndarray of ground-truth labels (or None)
        all_audio_names: list of audio names (or None)
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_audio_names = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating scores"):
            if len(batch) == 3:
                waveforms, lengths, labels = batch
                audio_names = None
            elif len(batch) == 4:
                waveforms, lengths, labels, audio_names = batch
            else:
                waveforms, lengths = batch[:2]
                labels = batch[2] if len(batch) > 2 else None
                audio_names = batch[3] if len(batch) > 3 else None

            waveforms = waveforms.to(device)
            lengths = lengths.to(device)

            logits = model(input_signal=waveforms, input_signal_length=lengths)
            probs = torch.softmax(logits, dim=1)[:, 0]  # bonafide probability

            all_probs.append(probs.cpu().numpy())
            if labels is not None:
                all_labels.append(
                    labels.numpy() if isinstance(labels, torch.Tensor) else labels
                )
            if audio_names is not None:
                all_audio_names.extend(audio_names)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels) if all_labels else None

    return all_probs, all_labels, all_audio_names if all_audio_names else None
