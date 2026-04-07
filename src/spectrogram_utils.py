import numpy as np
import torch


def amplitude_to_db_torch(magnitude: torch.Tensor, top_db: float = 80.0) -> torch.Tensor:
    magnitude = torch.clamp(magnitude, min=1e-10)
    ref_value = torch.clamp(torch.max(magnitude), min=1e-10)
    db = 20.0 * torch.log10(magnitude / ref_value)
    db = torch.clamp(db, min=-top_db)
    return db


def compute_log_spectrogram(
    y: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int | None = None,
    top_db: float = 80.0,
) -> np.ndarray:
    if win_length is None:
        win_length = n_fft

    waveform = torch.from_numpy(y.astype(np.float32))
    window = torch.hann_window(win_length)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = torch.abs(stft)
    db = amplitude_to_db_torch(magnitude, top_db=top_db)
    return db.cpu().numpy().astype(np.float32)
