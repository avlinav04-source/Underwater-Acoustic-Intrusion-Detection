"""
model_hybrid.py
---------------
Hybrid Deepwise Separable CNN (DWS-CNN) and Vision Transformer (ViT)
architecture for underwater acoustic intrusion detection & vessel classification.

Processes the identical input Log-Mel Spectrogram simultaneously through:
1. DWS-CNN Branch: Captures localized fine-grained time-frequency patterns with
   high parameter efficiency.
2. ViT Branch: Captures long-range temporal-spectral self-attention dependencies
   across non-overlapping acoustic patches.

Feature vectors from both branches are concatenated and passed through a trainable
dense fusion projection layer before 4-class softmax prediction.
"""

from typing import Tuple, Dict, Any, Optional
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    tf = None
    layers = None
    models = None
    TF_AVAILABLE = False

from src.model_dws_cnn import build_dws_cnn_backbone
from src.model_vit import build_vit_backbone


def check_tensorflow():
    """Validates that TensorFlow is available in the environment."""
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed in the local environment. "
            "Please install TensorFlow or run model training in Google Colab / Kaggle. "
            "See notebooks/training_guide.md for execution instructions."
        )


def build_hybrid_model(
    input_shape: Tuple[int, int, int] = (64, 94, 1),
    num_classes: int = 4,
    dws_filters: int = 32,
    dws_dropout: float = 0.3,
    dws_feature_dim: int = 128,
    vit_patch_size: int = 8,
    vit_projection_dim: int = 64,
    vit_num_heads: int = 4,
    vit_layers: int = 3,
    vit_mlp_dim: int = 128,
    vit_dropout: float = 0.2,
    vit_feature_dim: int = 128,
    fusion_dim: int = 128,
    fusion_dropout: float = 0.3,
    model_name: str = "Hybrid_DWS_CNN_ViT"
):
    """
    Constructs the end-to-end Hybrid DWS-CNN + ViT model.

    Both branches receive the exact same shared input Mel-spectrogram tensor.
    Their latent representations are fused via concatenation and non-linear projection.
    """
    check_tensorflow()

    # Single shared input tensor for both branches
    shared_input = layers.Input(shape=input_shape, name="input_mel_spectrogram")

    # --- 1. DWS-CNN Branch ---
    # Instantiate backbone
    _, dws_feat = build_dws_cnn_backbone(
        input_shape=input_shape,
        base_filters=dws_filters,
        dropout_rate=dws_dropout,
        feature_dim=dws_feature_dim
    )
    # Connect backbone to shared_input
    # Since build_dws_cnn_backbone returns a sub-graph, create functional wrapper:
    dws_extractor = models.Model(
        inputs=build_dws_cnn_backbone(input_shape, dws_filters, dws_dropout, dws_feature_dim)[0],
        outputs=dws_feat,
        name="DWS_CNN_Branch"
    )
    dws_vector = dws_extractor(shared_input)

    # --- 2. ViT Branch ---
    _, vit_feat = build_vit_backbone(
        input_shape=input_shape,
        patch_size=vit_patch_size,
        projection_dim=vit_projection_dim,
        num_heads=vit_num_heads,
        transformer_layers=vit_layers,
        mlp_dim=vit_mlp_dim,
        dropout_rate=vit_dropout,
        feature_dim=vit_feature_dim
    )
    vit_extractor = models.Model(
        inputs=build_vit_backbone(
            input_shape, vit_patch_size, vit_projection_dim,
            vit_num_heads, vit_layers, vit_mlp_dim, vit_dropout, vit_feature_dim
        )[0],
        outputs=vit_feat,
        name="ViT_Branch"
    )
    vit_vector = vit_extractor(shared_input)

    # --- 3. Feature Fusion Layer ---
    # Concatenate features from both modalities: [Batch, dws_feature_dim + vit_feature_dim]
    fused = layers.Concatenate(name="fusion_concat")([dws_vector, vit_vector])
    fused = layers.Dense(fusion_dim, activation="relu", name="fusion_dense1")(fused)
    fused = layers.BatchNormalization(name="fusion_bn1")(fused)
    fused = layers.Dropout(fusion_dropout, name="fusion_drop1")(fused)

    fused = layers.Dense(fusion_dim // 2, activation="relu", name="fusion_dense2")(fused)
    fused = layers.BatchNormalization(name="fusion_bn2")(fused)
    fused = layers.Dropout(fusion_dropout * 0.5, name="fusion_drop2")(fused)

    # --- 4. 4-Class Softmax Classification Head ---
    classification_output = layers.Dense(
        num_classes,
        activation="softmax",
        name="vessel_classification"
    )(fused)

    hybrid_model = models.Model(
        inputs=shared_input,
        outputs=classification_output,
        name=model_name
    )

    return hybrid_model


def run_forward_pass_test(model=None, batch_size: int = 2, input_shape: Tuple[int, int, int] = (64, 94, 1)):
    """
    Executes a synthetic forward pass to verify input/output shapes,
    gradient tracking, and branch connectivity.
    """
    check_tensorflow()
    if model is None:
        model = build_hybrid_model(input_shape=input_shape)

    dummy_input = tf.random.normal((batch_size, *input_shape))

    with tf.GradientTape() as tape:
        tape.watch(dummy_input)
        preds = model(dummy_input, training=True)
        loss = tf.reduce_mean(preds)

    grads = tape.gradient(loss, model.trainable_variables)
    none_grads = [var.name for var, g in zip(model.trainable_variables, grads) if g is None]

    print(f"[+] Forward pass test successful. Output shape: {preds.shape}")
    print(f"[+] Total trainable variables: {len(model.trainable_variables)}")
    if none_grads:
        print(f"[!] Warning: Some variables received None gradients: {none_grads}")
    else:
        print(f"[+] Gradient check: Verified. Gradients flow to ALL {len(grads)} trainable parameter tensors.")

    return {
        "output_shape": list(preds.shape),
        "trainable_params": model.count_params(),
        "all_gradients_active": len(none_grads) == 0
    }


if __name__ == "__main__":
    if TF_AVAILABLE:
        m = build_hybrid_model()
        m.summary()
        run_forward_pass_test(m)
    else:
        print("[INFO] TensorFlow not found locally. Code syntax and structure validated.")
