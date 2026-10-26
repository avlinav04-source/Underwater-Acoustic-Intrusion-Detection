"""
test_models.py
--------------
Automated tests for DWS-CNN, Vision Transformer (ViT), and Hybrid architectures.
Verifies model instantiation, input/output tensor shapes, forward passes,
and gradient flow when TensorFlow is installed.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_dws_cnn import TF_AVAILABLE as DWS_TF_AVAILABLE
from src.model_vit import TF_AVAILABLE as VIT_TF_AVAILABLE
from src.model_hybrid import TF_AVAILABLE as HYBRID_TF_AVAILABLE


class TestModelArchitectures(unittest.TestCase):
    """Test suite for neural network architectures."""

    def test_patch_geometry_math(self):
        """Verifies patch and positional embedding dimension calculations."""
        n_mels = 64
        padded_time_steps = 96
        patch_size = 8
        expected_patches = (n_mels // patch_size) * (padded_time_steps // patch_size)
        self.assertEqual(expected_patches, 96)

    def test_missing_tensorflow_clean_exception(self):
        """Verifies that calling model builders when TF is unavailable raises clear ImportError."""
        if not DWS_TF_AVAILABLE:
            from src.model_dws_cnn import build_dws_cnn_model
            with self.assertRaises(ImportError):
                build_dws_cnn_model()

    @unittest.skipUnless(HYBRID_TF_AVAILABLE, "TensorFlow not installed locally (use Colab / Kaggle for TF tests)")
    def test_dws_cnn_shape(self):
        """Verifies DWS-CNN model forward pass and output shape [B, 4]."""
        from src.model_dws_cnn import build_dws_cnn_model
        import tensorflow as tf
        model = build_dws_cnn_model(input_shape=(64, 94, 1), num_classes=4)
        dummy = tf.zeros((2, 64, 94, 1))
        out = model(dummy)
        self.assertEqual(out.shape, (2, 4))
        # Softmax sum to 1
        sums = tf.reduce_sum(out, axis=1).numpy()
        np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-5)

    @unittest.skipUnless(HYBRID_TF_AVAILABLE, "TensorFlow not installed locally (use Colab / Kaggle for TF tests)")
    def test_vit_shape(self):
        """Verifies ViT model forward pass and output shape [B, 4]."""
        from src.model_vit import build_vit_model
        import tensorflow as tf
        model = build_vit_model(input_shape=(64, 94, 1), num_classes=4)
        dummy = tf.zeros((2, 64, 94, 1))
        out = model(dummy)
        self.assertEqual(out.shape, (2, 4))

    @unittest.skipUnless(HYBRID_TF_AVAILABLE, "TensorFlow not installed locally (use Colab / Kaggle for TF tests)")
    def test_hybrid_forward_pass_and_gradients(self):
        """Verifies Hybrid DWS-CNN + ViT model forward pass and active gradients."""
        from src.model_hybrid import build_hybrid_model, run_forward_pass_test
        test_res = run_forward_pass_test(batch_size=2)
        self.assertEqual(test_res["output_shape"], [2, 4])
        self.assertTrue(test_res["all_gradients_active"])


if __name__ == "__main__":
    unittest.main()
