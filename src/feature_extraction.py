"""
feature_extraction.py
---------------------
Feature extraction pipeline for underwater acoustic recordings.

Transforms raw audio segments into 2D Log-Mel Spectrogram representations
suitable for feeding into Deepwise Separable CNN and Vision Transformer architectures.
Ensures standardized normalization, consistent tensor shapes, and safe numerical scaling.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import librosa


def compute_log_mel_spectrogram(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: Optional[float] = 8000.0,
    top_db: float = 80.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Computes a normalized log-Mel spectrogram from an audio waveform.

    Args:
        y: 1D audio waveform (numpy array of float32).
        sr: Sampling rate in Hz.
        n_fft: FFT window size.
        hop_length: Number of audio samples between adjacent STFT columns.
        n_mels: Number of Mel frequency bands.
        f_min: Lowest frequency in Hz.
        f_max: Highest frequency in Hz (defaults to sr // 2 if None).
        top_db: Maximum decibel threshold below peak power.
        normalize: Whether to min-max normalize spectrogram to [0.0, 1.0].

    Returns:
        np.ndarray of shape [n_mels, time_steps, 1] with dtype float32.
    """
    if f_max is None:
        f_max = float(sr // 2)

    # Compute power Mel-spectrogram
    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0
    )

    # Convert to log scale (decibels) relative to max power
    log_mel = librosa.power_to_db(mel_power, ref=np.max, top_db=top_db)

    # Min-Max Normalization to [0.0, 1.0] for stable neural network training
    if normalize:
        min_val = -top_db
        max_val = 0.0
        # Clip to [-top_db, 0.0] and rescale
        clipped = np.clip(log_mel, min_val, max_val)
        norm_mel = (clipped - min_val) / (max_val - min_val + 1e-8)
    else:
        norm_mel = log_mel

    # Add single channel dimension: [n_mels, time_steps, 1]
    norm_mel = np.expand_dims(norm_mel.astype(np.float32), axis=-1)

    return norm_mel


def extract_features_from_segments(
    segments: List[np.ndarray],
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: Optional[float] = 8000.0,
    top_db: float = 80.0
) -> np.ndarray:
    """
    Extracts log-Mel spectrograms for a list of audio segments.

    Returns:
        np.ndarray of shape [N, n_mels, time_steps, 1].
    """
    if not segments:
        return np.empty((0, n_mels, 0, 1), dtype=np.float32)

    specs = [
        compute_log_mel_spectrogram(
            seg, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max, top_db=top_db
        )
        for seg in segments
    ]
    return np.array(specs, dtype=np.float32)


def extract_features_for_recording(
    audio_path: str or Path,
    sr: int = 16000,
    segment_duration: float = 3.0,
    min_duration: float = 3.0,
    normalize_peak: float = 0.95,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: Optional[float] = 8000.0,
    top_db: float = 80.0
) -> np.ndarray:
    """
    Loads, normalizes, windows, and extracts log-Mel spectrograms for a single recording.
    
    Returns:
        np.ndarray of shape [N_segments, n_mels, time_steps, 1].
    """
    from src.audio_preprocessing import load_audio, normalize_audio, segment_audio

    y, loaded_sr = load_audio(audio_path, target_sr=sr, mono=True)
    y_norm = normalize_audio(y, peak=normalize_peak)
    segments = segment_audio(y_norm, sr=sr, segment_duration=segment_duration, min_duration=min_duration)
    if not segments:
        return np.empty((0, n_mels, 0, 1), dtype=np.float32)

    return extract_features_from_segments(
        segments, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, f_min=f_min, f_max=f_max, top_db=top_db
    )


def extract_features_for_split(
    records: List[Dict[str, Any]],
    raw_data_dir: str or Path,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts all segment features and labels for a specific split (train, val, or test).

    Returns:
        X: np.ndarray [Total_segments, n_mels, time_steps, 1]
        y: np.ndarray [Total_segments] integer class indices
        rec_ids: List of original recording identifiers for each segment
    """
    from src.utils import CLASS_TO_IDX

    audio_cfg = config.get("audio", {})
    feat_cfg = config.get("features", {})

    sr = audio_cfg.get("target_sample_rate", 16000)
    seg_dur = audio_cfg.get("segment_duration", 3.0)
    min_dur = audio_cfg.get("min_duration", 3.0)
    norm_peak = audio_cfg.get("normalize_peak", 0.95)

    n_fft = feat_cfg.get("n_fft", 1024)
    hop_length = feat_cfg.get("hop_length", 512)
    n_mels = feat_cfg.get("n_mels", 64)
    f_min = feat_cfg.get("f_min", 20.0)
    f_max = feat_cfg.get("f_max", 8000.0)
    top_db = feat_cfg.get("top_db", 80.0)

    all_specs = []
    all_labels = []
    all_rec_ids = []

    root_dir = Path(raw_data_dir)

    for rec in records:
        cls_name = rec["class_name"]
        if cls_name not in CLASS_TO_IDX:
            continue
        label_idx = CLASS_TO_IDX[cls_name]

        # Resolve full audio path
        rel_path = rec.get("relative_to_data_root")
        if rel_path:
            full_audio_path = root_dir / rel_path
        else:
            full_audio_path = Path(rec.get("relative_path", ""))

        if not full_audio_path.is_file():
            print(f"  [!] Warning: file not found: {full_audio_path}")
            continue

        try:
            specs = extract_features_for_recording(
                full_audio_path,
                sr=sr,
                segment_duration=seg_dur,
                min_duration=min_dur,
                normalize_peak=norm_peak,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                top_db=top_db
            )
            n_segs = len(specs)
            if n_segs > 0:
                all_specs.append(specs)
                all_labels.extend([label_idx] * n_segs)
                rec_id = str(rel_path or full_audio_path.name)
                all_rec_ids.extend([rec_id] * n_segs)
        except Exception as e:
            print(f"  [!] Error extracting features from {full_audio_path.name}: {e}")

    if all_specs:
        X = np.concatenate(all_specs, axis=0)
        y = np.array(all_labels, dtype=np.int32)
    else:
        X = np.empty((0, n_mels, 94, 1), dtype=np.float32)
        y = np.empty((0,), dtype=np.int32)

    return X, y, all_rec_ids
