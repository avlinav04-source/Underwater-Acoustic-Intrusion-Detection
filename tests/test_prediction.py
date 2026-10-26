"""
test_prediction.py
------------------
Automated tests for prediction pipeline, missing checkpoint handling,
short audio validation, and illustrative security rule evaluation.
"""

import unittest
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import VesselPredictor
from src.utils import CANONICAL_CLASSES


class TestPredictionPipeline(unittest.TestCase):
    """Test suite for prediction logic and edge case handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create valid 4-second audio file (16000 Hz)
        sr = 16000
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 350 * t).astype(np.float32)

        self.valid_wav = self.temp_path / "valid_4s.wav"
        sf.write(str(self.valid_wav), audio, sr)

        # Create too-short audio file (1.0 second < 3.0 second segment duration)
        short_audio = audio[: int(sr * 1.0)]
        self.short_wav = self.temp_path / "short_1s.wav"
        sf.write(str(self.short_wav), short_audio, sr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_checkpoint_handling(self):
        """Verifies that missing model checkpoint returns a clean error dict without crashing."""
        non_existent_model = self.temp_path / "non_existent_checkpoint.keras"
        predictor = VesselPredictor(model_path=non_existent_model)
        self.assertFalse(predictor.is_model_ready())

        result = predictor.predict(self.valid_wav)
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_nonexistent_audio_file(self):
        """Verifies graceful error handling for missing audio files."""
        fake_path = self.temp_path / "ghost.wav"
        predictor = VesselPredictor()
        result = predictor.predict(fake_path)
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_predictor_audio_preprocessing_shape(self):
        """Verifies that preprocess_audio_file generates correct shape [N, 64, 94, 1]."""
        predictor = VesselPredictor()
        features, duration = predictor.preprocess_audio_file(self.valid_wav)
        self.assertEqual(duration, 4.0)
        # 4s audio produces 1 3s segment
        self.assertEqual(features.shape, (1, 64, 94, 1))

    def test_short_audio_rejection(self):
        """Verifies that audio shorter than min_duration is rejected with a clear message."""
        predictor = VesselPredictor()
        with self.assertRaises(ValueError):
            predictor.preprocess_audio_file(self.short_wav)

    def test_illustrative_security_rule_structure(self):
        """Verifies illustrative policy returns correct fields and academic clarification."""
        predictor = VesselPredictor()
        rule = predictor._apply_illustrative_rule("Cargo")
        self.assertIn("zone_alert_status", rule)
        self.assertIn("academic_clarification", rule)
        self.assertEqual(rule["zone_alert_status"], "FLAGGED_FOR_INSPECTION")


if __name__ == "__main__":
    unittest.main()
