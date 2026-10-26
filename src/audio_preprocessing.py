"""
audio_preprocessing.py
----------------------
Reproducible audio preprocessing pipeline for underwater acoustic recordings.

Provides safe audio loading, mono conversion, high-quality resampling,
robust peak normalization, segment windowing with short-audio handling,
and waveform / spectrogram visualization.
"""

import math
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt


class AudioLoadingError(Exception):
    """Raised when an audio file cannot be loaded or is corrupted."""
    pass


def load_audio(
    file_path: str or Path,
    target_sr: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Safely loads an audio file, converts to mono, and resamples to target_sr.

    Args:
        file_path: Path to the audio file.
        target_sr: Target sampling rate in Hz.
        mono: Whether to downmix multi-channel audio to mono.

    Returns:
        Tuple of (audio_waveform as float32 np.ndarray, sample_rate as int).
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {p.resolve()}")

    if p.stat().st_size == 0:
        raise AudioLoadingError(f"Audio file is empty (0 bytes): {p.name}")

    try:
        # Load audio data using soundfile first (faster, reliable)
        data, native_sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception as sf_err:
        try:
            # Fallback to librosa
            data, native_sr = librosa.load(str(p), sr=None, mono=mono)
        except Exception as lb_err:
            raise AudioLoadingError(
                f"Failed to read audio file '{p.name}'. soundfile error: {sf_err}; librosa error: {lb_err}"
            )

    # Convert to mono if multi-channel
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if native_sr != target_sr:
        try:
            data = librosa.resample(data, orig_sr=native_sr, target_sr=target_sr, res_type="soxr_hq")
        except Exception:
            try:
                data = librosa.resample(data, orig_sr=native_sr, target_sr=target_sr, res_type="scipy")
            except Exception:
                data = librosa.resample(data, orig_sr=native_sr, target_sr=target_sr)

    # Ensure float32 array
    data = data.astype(np.float32)

    return data, target_sr


def normalize_audio(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """
    Safely normalizes the audio signal peak amplitude to a target value.
    Prevents division-by-zero on silent signals.

    Args:
        y: 1D audio waveform array.
        peak: Target maximum peak amplitude (0.0 < peak <= 1.0).

    Returns:
        Normalized float32 audio waveform.
    """
    if y.size == 0:
        return y

    max_val = np.max(np.abs(y))
    if max_val > 1e-7:
        return (y / max_val * peak).astype(np.float32)
    # Signal is essentially pure silence
    return np.zeros_like(y, dtype=np.float32)


def segment_audio(
    y: np.ndarray,
    sr: int,
    segment_duration: float = 3.0,
    hop_duration: Optional[float] = None,
    min_duration: float = 3.0
) -> List[np.ndarray]:
    """
    Segments a continuous audio signal into fixed-length windows.

    Args:
        y: 1D audio waveform.
        sr: Audio sample rate in Hz.
        segment_duration: Desired segment length in seconds.
        hop_duration: Hop step between segment starts (defaults to non-overlapping = segment_duration).
        min_duration: Minimum required signal duration in seconds to generate segments.

    Returns:
        List of 1D numpy arrays, each of exact length (sr * segment_duration).
    """
    target_samples = int(round(segment_duration * sr))
    total_samples = len(y)

    # Check minimum length requirement
    if total_samples < int(round(min_duration * sr)):
        return []

    # If shorter than target_samples, pad to target_samples
    if total_samples < target_samples:
        pad_amount = target_samples - total_samples
        padded = np.pad(y, (0, pad_amount), mode="constant", constant_values=0.0)
        return [padded.astype(np.float32)]

    hop_samples = int(round((hop_duration or segment_duration) * sr))
    segments = []

    for start in range(0, total_samples - target_samples + 1, hop_samples):
        chunk = y[start : start + target_samples]
        if len(chunk) == target_samples:
            segments.append(chunk.astype(np.float32))

    return segments


def visualize_audio_sample(
    y: np.ndarray,
    sr: int,
    mel_spec: Optional[np.ndarray] = None,
    title: str = "Underwater Audio Sample",
    save_path: Optional[str or Path] = None
) -> None:
    """
    Plots the waveform and Mel-spectrogram side-by-side.
    """
    num_subplots = 2 if mel_spec is not None else 1
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3.5 * num_subplots), sharex=False)
    if num_subplots == 1:
        axes = [axes]

    # Waveform plot
    time_axis = np.linspace(0, len(y) / sr, num=len(y))
    axes[0].plot(time_axis, y, color="#1f77b4", alpha=0.85, linewidth=0.7)
    axes[0].set_title(f"{title} - Waveform", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Mel-spectrogram plot
    if mel_spec is not None:
        # mel_spec shape: (n_mels, time_steps)
        if mel_spec.ndim == 3 and mel_spec.shape[-1] == 1:
            mel_plot = mel_spec[:, :, 0]
        else:
            mel_plot = mel_spec
        img = axes[1].imshow(
            mel_plot,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, len(y) / sr, 0, sr // 2]
        )
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB", label="Log-Mel Power (dB)")
        axes[1].set_title(f"{title} - Log-Mel Spectrogram", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Frequency (Hz)")

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=150)
        plt.close(fig)
    else:
        plt.show()
