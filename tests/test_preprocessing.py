"""
test_preprocessing.py
---------------------
Automated tests for audio loading, amplitude normalization,
audio window segmentation, and log-Mel spectrogram feature extraction.
"""

import unittest
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_preprocessing import (
    load_audio,
    normalize_audio,
    segment_audio,
    AudioLoadingError
)
from src.feature_extraction import (
    compute_log_mel_spectrogram,
    extract_features_from_segments
)


class TestAudioPreprocessing(unittest.TestCase):
    """Test suite for preprocessing functions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.sr = 16000
        self.duration = 6.0  # 6 seconds -> should produce 2 non-overlapping 3s segments
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.test_audio = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

        self.wav_path = self.temp_path / "sample_6s.wav"
        sf.write(str(self.wav_path), self.test_audio, self.sr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_audio_success(self):
        """Verifies safe loading and resampling of a WAV file."""
        data, sr = load_audio(self.wav_path, target_sr=16000, mono=True)
        self.assertEqual(sr, 16000)
        self.assertEqual(len(data), int(16000 * 6.0))
        self.assertEqual(data.dtype, np.float32)

    def test_load_audio_empty_file(self):
        """Verifies that attempting to load an empty file raises AudioLoadingError."""
        empty_path = self.temp_path / "empty.wav"
        empty_path.touch()
        with self.assertRaises(AudioLoadingError):
            load_audio(empty_path)

    def test_normalize_audio_regular(self):
        """Verifies peak amplitude normalization."""
        unscaled = np.array([-2.0, 0.5, 4.0], dtype=np.float32)
        normalized = normalize_audio(unscaled, peak=0.95)
        self.assertAlmostEqual(np.max(np.abs(normalized)), 0.95, places=5)

    def test_normalize_audio_silence(self):
        """Verifies that purely silent audio does not trigger division-by-zero."""
        silent = np.zeros(1000, dtype=np.float32)
        result = normalize_audio(silent, peak=0.95)
        self.assertEqual(np.max(np.abs(result)), 0.0)
        self.assertEqual(len(result), 1000)

    def test_segment_audio_exact_windows(self):
        """Verifies segmentation of 6-second audio into two 3-second segments."""
        segments = segment_audio(self.test_audio, sr=16000, segment_duration=3.0)
        self.assertEqual(len(segments), 2)
        for seg in segments:
            self.assertEqual(len(seg), int(16000 * 3.0))

    def test_segment_audio_short_with_padding(self):
        """Verifies that audio meeting min_duration is padded to target length."""
        short_audio = self.test_audio[: int(16000 * 2.0)]  # 2.0 seconds
        segments = segment_audio(
            short_audio, sr=16000, segment_duration=3.0, min_duration=1.5
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), int(16000 * 3.0))

    def test_log_mel_spectrogram_shape_and_bounds(self):
        """Verifies log-Mel spectrogram shape [n_mels, time_steps, 1] and value bounds [0, 1]."""
        segment = self.test_audio[: int(16000 * 3.0)]
        mel = compute_log_mel_spectrogram(
            segment, sr=16000, n_fft=1024, hop_length=512, n_mels=64, top_db=80.0
        )
        self.assertEqual(mel.shape, (64, 94, 1))
        self.assertGreaterEqual(mel.min(), 0.0)
        self.assertLessEqual(mel.max(), 1.0)

    def test_batch_extract_features(self):
        """Verifies batch extraction returns tensor of shape [N, 64, 94, 1]."""
        segments = segment_audio(self.test_audio, sr=16000, segment_duration=3.0)
        features = extract_features_from_segments(segments, sr=16000, n_mels=64)
        self.assertEqual(features.shape, (2, 64, 94, 1))


if __name__ == "__main__":
    unittest.main()
