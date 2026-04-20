from pathlib import Path

import numpy as np
import torch

from audio_utils import load_audio_ffmpeg
from model_factory import create_model
from spectrogram_utils import compute_log_spectrogram


def load_checkpoint_model(model_name: str, checkpoint_path: Path, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(model_name)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, torch.device(device)


def make_crops(spec: np.ndarray, crop_width: int = 64, stride: int = 32) -> list[np.ndarray]:
    total_width = int(spec.shape[1])
    if total_width < crop_width:
        pad_width = crop_width - total_width
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="edge")
        total_width = int(spec.shape[1])

    crops = []
    for start in range(0, total_width - crop_width + 1, stride):
        crops.append(spec[:, start:start + crop_width])
    return crops


def predict_spectrogram(
    model,
    device: torch.device,
    spec: np.ndarray,
    crop_width: int = 64,
    stride: int = 32,
):
    crops = make_crops(spec, crop_width=crop_width, stride=stride)
    if not crops:
        raise ValueError("Tahmin icin hic crop olusmadi.")

    batch = np.stack(crops, axis=0)
    batch = np.expand_dims(batch, axis=1)
    tensor = torch.tensor(batch, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    track_prob = float(np.mean(probs))
    track_pred = int(track_prob >= 0.5)

    return {
        "track_prob_fake": track_prob,
        "track_pred": track_pred,
        "n_crops": len(crops),
        "crop_probs": probs.tolist(),
        "spectrogram": spec,
    }


def predict_audio_path(
    model,
    device: torch.device,
    audio_path: Path,
    sample_rate: int = 16000,
    clip_seconds: int = 10,
    n_fft: int = 1024,
    hop_length: int = 320,
    crop_width: int = 64,
    stride: int = 32,
):
    waveform = load_audio_ffmpeg(audio_path, sample_rate=sample_rate, clip_seconds=clip_seconds)
    spec = compute_log_spectrogram(waveform, n_fft=n_fft, hop_length=hop_length)
    prediction = predict_spectrogram(
        model,
        device,
        spec,
        crop_width=crop_width,
        stride=stride,
    )
    prediction["audio_path"] = str(audio_path)
    return prediction
