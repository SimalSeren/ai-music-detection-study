import os
import subprocess
from pathlib import Path

import numpy as np


DEFAULT_FFMPEG_PATHS = [
    Path(
        r"C:\Users\simalseren\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    ),
    Path(
        r"C:\Users\simalseren\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
    ),
]


def resolve_ffmpeg() -> str:
    env_path = os.environ.get("FFMPEG_EXE")
    if env_path and Path(env_path).exists():
        return env_path

    for candidate in DEFAULT_FFMPEG_PATHS:
        if candidate.exists():
            return str(candidate)

    return "ffmpeg"


def load_audio_ffmpeg(audio_path: Path, sample_rate: int, clip_seconds: int) -> np.ndarray:
    ffmpeg_exe = resolve_ffmpeg()
    max_len = sample_rate * clip_seconds

    command = [
        ffmpeg_exe,
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-t",
        str(clip_seconds),
        "pipe:1",
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg decode failed for {audio_path}: {stderr.strip()}")

    y = np.frombuffer(process.stdout, dtype=np.float32)
    y = y[:max_len]
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    return y
