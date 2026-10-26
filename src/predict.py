"""
predict.py
----------
Inference engine for Underwater Acoustic Vessel Classification.

Loads the trained model checkpoint, applies the exact same preprocessing pipeline
used during training, performs windowed segment classification, and aggregates
predictions to return overall vessel class and confidence distribution.
Includes an explicit, clearly segregated illustrative security demonstration rule.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config,
    load_json,
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)
from src.audio_preprocessing import load_audio, normalize_audio, segment_audio, AudioLoadingError
from src.feature_extraction import extract_features_from_segments

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    tf = None
    load_model = None
    TF_AVAILABLE = False


class VesselPredictor:
    """
    Manages loading the trained hybrid acoustic model and running inference on audio files.
    """

    def __init__(self, model_path: Optional[str or Path] = None, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.audio_cfg = self.config.get("audio", {})
        self.feat_cfg = self.config.get("features", {})
        self.paths_cfg = self.config.get("paths", {})

        # Default model checkpoint path
        default_model = PROJECT_ROOT / self.paths_cfg.get("models_dir", "models") / "best_hybrid_model.keras"
        self.model_path = Path(model_path or default_model)

        self.model = None
        self._load_model_if_available()

    def _load_model_if_available(self):
        """Loads trained weights if TensorFlow and checkpoint are present."""
        if not TF_AVAILABLE:
            return

        if self.model_path.is_file():
            try:
                from src.model_vit import PatchAndPositionEmbedding
                custom_objects = {"PatchAndPositionEmbedding": PatchAndPositionEmbedding}
                self.model = load_model(str(self.model_path), custom_objects=custom_objects)
                print(f"[+] Loaded model checkpoint from: {self.model_path.resolve()}")
            except Exception as e:
                print(f"[!] Error loading model checkpoint '{self.model_path}': {e}")
                self.model = None

    def is_model_ready(self) -> bool:
        """Checks if a trained model is ready for inference."""
        return self.model is not None

    def preprocess_audio_file(self, audio_path: str or Path) -> Tuple[np.ndarray, float]:
        """
        Loads, resamples, normalizes, windows, and extracts log-Mel spectrograms.
        Uses exact settings from training.

        Returns:
            Tuple of (features_array of shape [N, 64, 94, 1], audio_duration_seconds).
        """
        sr = self.audio_cfg.get("target_sample_rate", 16000)
        seg_dur = self.audio_cfg.get("segment_duration", 3.0)
        min_dur = self.audio_cfg.get("min_duration", 3.0)
        norm_peak = self.audio_cfg.get("normalize_peak", 0.95)

        y, loaded_sr = load_audio(audio_path, target_sr=sr, mono=True)
        duration_sec = float(len(y) / sr)

        y_norm = normalize_audio(y, peak=norm_peak)
        segments = segment_audio(y_norm, sr=sr, segment_duration=seg_dur, min_duration=min_dur)

        if not segments:
            raise ValueError(
                f"Audio file is too short ({duration_sec:.2f}s). "
                f"Minimum required duration is {min_dur:.1f}s."
            )

        features = extract_features_from_segments(
            segments,
            sr=sr,
            n_fft=self.feat_cfg.get("n_fft", 1024),
            hop_length=self.feat_cfg.get("hop_length", 512),
            n_mels=self.feat_cfg.get("n_mels", 64),
            f_min=self.feat_cfg.get("f_min", 20.0),
            f_max=self.feat_cfg.get("f_max", 8000.0),
            top_db=self.feat_cfg.get("top_db", 80.0)
        )

        return features, duration_sec

    def predict(
        self,
        audio_path: str or Path,
        hypothetical_security_policy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs vessel classification on an audio recording.

        Args:
            audio_path: Path to WAV audio file.
            hypothetical_security_policy: Optional illustrative rule check.

        Returns:
            Dict containing predicted class, confidence, class probabilities,
            audio duration, number of segments, and security alert demonstration note.
        """
        if not TF_AVAILABLE:
            return {
                "success": False,
                "error": "TensorFlow is not installed in the local environment. Please run training & inference in Google Colab or install TensorFlow.",
                "model_available": False,
            }

        if not self.is_model_ready():
            return {
                "success": False,
                "error": f"No trained model found at '{self.model_path.resolve()}'. Please train a model first using 'src/train.py'.",
                "model_available": False,
            }

        try:
            features, duration = self.preprocess_audio_file(audio_path)
        except Exception as e:
            return {
                "success": False,
                "error": f"Audio processing failed: {str(e)}",
                "model_available": True,
            }

        # Predict across all segments
        segment_probs = self.model.predict(features, verbose=0)
        # Aggregate across segments via mean pooling
        mean_probs = np.mean(segment_probs, axis=0)

        pred_idx = int(np.argmax(mean_probs))
        pred_class = CANONICAL_CLASSES[pred_idx]
        pred_conf = float(mean_probs[pred_idx])

        class_probabilities = {
            cls_name: round(float(mean_probs[i]), 4)
            for i, cls_name in enumerate(CANONICAL_CLASSES)
        }

        # Illustrative Demonstration Rule (Honest Security Context)
        # Note: Vessel type classification != intrusion detection.
        # This rule demonstrates how an operational system would apply zone rules.
        illustrative_rule = self._apply_illustrative_rule(pred_class, hypothetical_security_policy)

        return {
            "success": True,
            "predicted_class": pred_class,
            "confidence": round(pred_conf, 4),
            "confidence_percentage": f"{pred_conf * 100:.1f}%",
            "class_probabilities": class_probabilities,
            "audio_duration_seconds": round(duration, 2),
            "num_segments_analyzed": len(features),
            "model_checkpoint": self.model_path.name,
            "illustrative_security_rule": illustrative_rule,
        }

    def _apply_illustrative_rule(self, pred_class: str, policy: Optional[str] = None) -> Dict[str, Any]:
        """
        Applies a simulated illustrative security check.
        Explicitly clarifies that vessel category is not an intrinsic intrusion label.
        """
        policy_name = policy or "Marine Protected Area (Illustrative Demo Policy)"

        # Example demonstration rule: In an MPA, heavy industrial vessels trigger alerts
        is_alert = pred_class in ["Cargo", "Tanker"]
        explanation = (
            f"Under the demo policy '{policy_name}', {pred_class} vessels are flagged for zone clearance verification. "
            "Note: This is a configurable illustrative policy rule, NOT a learned intrusion detector. "
            "Real intrusion detection requires vessel AIS transponder verification and authorization logs."
        )

        return {
            "policy_applied": policy_name,
            "zone_alert_status": "FLAGGED_FOR_INSPECTION" if is_alert else "PERMITTED_CORRIDOR",
            "academic_clarification": explanation,
        }


def predict_audio_cli(audio_path: str, model_path: Optional[str] = None):
    """CLI runner for prediction."""
    predictor = VesselPredictor(model_path=model_path)
    res = predictor.predict(audio_path)

    print("\n" + "=" * 65)
    print(" UNDERWATER ACOUSTIC VESSEL CLASSIFICATION RESULT")
    print("=" * 65)
    if not res.get("success"):
        print(f"[ERROR] {res.get('error')}")
        print("=" * 65 + "\n")
        return

    print(f"File Analyzed        : {Path(audio_path).name}")
    print(f"Duration             : {res['audio_duration_seconds']}s ({res['num_segments_analyzed']} segments)")
    print(f"Predicted Category   : {res['predicted_class']} (Confidence: {res['confidence_percentage']})")
    print("-" * 65)
    print("Class Probability Breakdown:")
    for cls_name, prob in res["class_probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  * {cls_name:<16} : {prob*100:>5.1f}% | {bar}")
    print("-" * 65)
    rule = res["illustrative_security_rule"]
    print(f"Illustrative Security Check : [{rule['zone_alert_status']}]")
    print(f"Note: {rule['academic_clarification']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict vessel type from underwater audio.")
    parser.add_argument("audio_path", type=str, help="Path to input WAV file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to .keras model checkpoint")
    args = parser.parse_args()

    predict_audio_cli(args.audio_path, args.model_path)
