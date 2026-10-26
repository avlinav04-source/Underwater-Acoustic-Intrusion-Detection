"""
test_dataset.py
---------------
Automated tests for dataset path validation, audio file inspection,
corrupted file handling, and label mappings.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import soundfile as sf
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import CANONICAL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, load_config
from src.analyze_audio import inspect_audio_file, scan_and_analyze_dataset, generate_analysis_summary


class TestDatasetInspection(unittest.TestCase):
    """Test suite for dataset inspection and audio analysis utilities."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a synthetic valid WAV file (1 sec, 16000 Hz, mono)
        self.sr = 16000
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        self.valid_wav_path = self.temp_path / "Cargo" / "test_valid.wav"
        self.valid_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(self.valid_wav_path), waveform, self.sr)

        # Create an empty file (0 bytes)
        self.empty_file_path = self.temp_path / "Tanker" / "test_empty.wav"
        self.empty_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.empty_file_path.touch()

        # Create a corrupted audio file
        self.corrupt_file_path = self.temp_path / "Passengership" / "test_corrupt.wav"
        self.corrupt_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corrupt_file_path, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00invalid-audio-junk")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_canonical_classes_and_mappings(self):
        """Verifies that the canonical vessel categories and index mappings match requirements."""
        expected = ["Cargo", "Passengership", "Tanker", "Tug"]
        self.assertEqual(CANONICAL_CLASSES, expected)
        self.assertEqual(len(CLASS_TO_IDX), 4)
        for cls_name in expected:
            idx = CLASS_TO_IDX[cls_name]
            self.assertEqual(IDX_TO_CLASS[idx], cls_name)

    def test_valid_audio_inspection(self):
        """Verifies inspection of a valid WAV audio file."""
        info = inspect_audio_file(self.valid_wav_path, "Cargo")
        self.assertTrue(info["is_valid"])
        self.assertEqual(info["sample_rate"], 16000)
        self.assertEqual(info["channels"], 1)
        self.assertAlmostEqual(info["duration_seconds"], 1.0, places=1)
        self.assertEqual(info["class_name"], "Cargo")
        self.assertIsNone(info["error_message"])

    def test_empty_audio_handling(self):
        """Verifies safe handling of empty audio files without unhandled exceptions."""
        info = inspect_audio_file(self.empty_file_path, "Tanker")
        self.assertFalse(info["is_valid"])
        self.assertEqual(info["duration_seconds"], 0.0)
        self.assertIn("Empty", info["error_message"])

    def test_corrupted_audio_handling(self):
        """Verifies safe detection and error reporting for corrupted audio files."""
        info = inspect_audio_file(self.corrupt_file_path, "Passengership")
        self.assertFalse(info["is_valid"])
        self.assertIsNotNone(info["error_message"])

    def test_dataset_scan_and_summary(self):
        """Verifies directory scanning and summary metrics generation."""
        # Add Tug folder so all 4 exist
        (self.temp_path / "Tug").mkdir(parents=True, exist_ok=True)

        df = scan_and_analyze_dataset(self.temp_path)
        self.assertEqual(len(df), 3)  # valid, empty, corrupt
        self.assertEqual(df["is_valid"].sum(), 1)

        summary = generate_analysis_summary(df)
        self.assertEqual(summary["total_recordings_scanned"], 3)
        self.assertEqual(summary["valid_recordings"], 1)
        self.assertEqual(summary["corrupted_or_empty"], 2)


class TestDatasetSplit(unittest.TestCase):
    """Test suite for recording-level data splitting and leakage prevention."""

    def setUp(self):
        # Create a mock metadata dataframe mirroring the 63 DeepShip recordings
        records = []
        counts = {"Cargo": 12, "Passengership": 20, "Tanker": 28, "Tug": 3}
        for cls_name, count in counts.items():
            for i in range(count):
                records.append({
                    "file_name": f"{i}.wav",
                    "relative_to_data_root": f"{cls_name}\\{i}.wav",
                    "class_name": cls_name,
                    "is_valid": True,
                    "duration_seconds": 60.0
                })
        self.mock_df = pd.DataFrame(records)

    def test_zero_recording_leakage(self):
        """Verifies that no recording ID is shared across train, val, or test splits."""
        from src.dataset_split import create_recording_level_split
        manifest = create_recording_level_split(self.mock_df, seed=42)

        def get_ids(records):
            return set(r["relative_to_data_root"] for r in records)

        train_ids = get_ids(manifest["splits"]["train"])
        val_ids = get_ids(manifest["splits"]["val"])
        test_ids = get_ids(manifest["splits"]["test"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

        # Total recordings must equal sum of split recordings
        self.assertEqual(len(train_ids) + len(val_ids) + len(test_ids), 63)

    def test_tug_scarce_class_distribution(self):
        """Verifies Tug with 3 recordings is distributed with 1 in each split."""
        from src.dataset_split import create_recording_level_split
        manifest = create_recording_level_split(self.mock_df, seed=42)
        tug_audit = manifest["class_audit"]["Tug"]
        self.assertEqual(tug_audit["train_recordings"], 1)
        self.assertEqual(tug_audit["val_recordings"], 1)
        self.assertEqual(tug_audit["test_recordings"], 1)

    def test_class_weights_balance_minority(self):
        """Verifies that computed training class weights prioritize the minority class."""
        from src.dataset_split import create_recording_level_split
        manifest = create_recording_level_split(self.mock_df, seed=42)
        weights = manifest["class_weights"]
        self.assertEqual(len(weights), 4)
        # Tug index is 3, should have the highest weight
        self.assertGreater(weights[3], weights[0])
        self.assertGreater(weights[3], weights[1])
        self.assertGreater(weights[3], weights[2])


if __name__ == "__main__":
    import pandas as pd
    unittest.main()
