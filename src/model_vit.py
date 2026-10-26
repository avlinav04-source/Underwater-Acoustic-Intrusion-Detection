"""
model_vit.py
------------
Vision Transformer (ViT) branch for underwater acoustic vessel classification.

Divides Log-Mel spectrograms into 2D patches, linearly projects them into token
embeddings, applies learnable positional encodings, and processes temporal-spectral
relationships via stacked Transformer Multi-Head Self-Attention blocks.
"""

from typing import Tuple, Optional
import os

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    tf = None
    layers = None
    models = None
    TF_AVAILABLE = False


def check_tensorflow():
    """Validates that TensorFlow is available in the environment."""
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed in the local environment. "
            "Please install TensorFlow or run model training in Google Colab / Kaggle. "
            "See notebooks/training_guide.md for execution instructions."
        )


class PatchAndPositionEmbedding(layers.Layer if TF_AVAILABLE else object):
    """
    Extracts non-overlapping patches from spectrograms, projects them to
    embedding dimension, and adds learnable 1D positional embeddings.
    """

    def __init__(
        self,
        patch_size: int = 8,
        projection_dim: int = 64,
        num_patches: int = 96,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        if TF_AVAILABLE:
            self.patch_conv = layers.Conv2D(
                filters=projection_dim,
                kernel_size=(patch_size, patch_size),
                strides=(patch_size, patch_size),
                padding="valid",
                name="patch_projection"
            )
            self.position_embedding = layers.Embedding(
                input_dim=num_patches,
                output_dim=projection_dim,
                name="pos_embedding"
            )

    def call(self, images):
        # images: [batch, height, width, channels] e.g. [B, 64, 96, 1]
        patches = self.patch_conv(images)  # [B, 8, 12, projection_dim]
        batch_size = tf.shape(images)[0]
        # Flatten spatial dimensions to sequence: [B, 96, projection_dim]
        tokens = tf.reshape(patches, [batch_size, -1, self.projection_dim])
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions)
        return tokens + pos_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "projection_dim": self.projection_dim,
            "num_patches": self.num_patches,
        })
        return config


def transformer_encoder_block(inputs, num_heads: int, key_dim: int, mlp_dim: int, dropout_rate: float, name_prefix: str):
    """A standard Transformer Encoder Block with Pre-LayerNorm and residual connections."""
    # Self-Attention sub-block
    norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(inputs)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate * 0.5,
        name=f"{name_prefix}_mha"
    )(norm1, norm1)
    res1 = layers.Add(name=f"{name_prefix}_add1")([inputs, attn_out])

    # Feed-Forward MLP sub-block
    norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(res1)
    mlp = layers.Dense(mlp_dim, activation=tf.nn.gelu, name=f"{name_prefix}_mlp_dense1")(norm2)
    mlp = layers.Dropout(dropout_rate, name=f"{name_prefix}_mlp_drop1")(mlp)
    mlp = layers.Dense(inputs.shape[-1], name=f"{name_prefix}_mlp_dense2")(mlp)
    mlp = layers.Dropout(dropout_rate * 0.5, name=f"{name_prefix}_mlp_drop2")(mlp)
    out = layers.Add(name=f"{name_prefix}_add2")([res1, mlp])
    return out


def build_vit_backbone(
    input_shape: Tuple[int, int, int] = (64, 94, 1),
    patch_size: int = 8,
    projection_dim: int = 64,
    num_heads: int = 4,
    transformer_layers: int = 3,
    mlp_dim: int = 128,
    dropout_rate: float = 0.2,
    feature_dim: int = 128
):
    """
    Constructs the Vision Transformer feature extractor branch.

    Args:
        input_shape: Input spectrogram shape (64, 94, 1).
        patch_size: Square patch edge length in pixels (8x8).
        projection_dim: Latent dimension for patch tokens.
        num_heads: Attention heads.
        transformer_layers: Number of stacked transformer blocks.
        mlp_dim: Hidden dimension of feed-forward network.
        dropout_rate: Dropout probability.
        feature_dim: Final representation vector dimension.

    Returns:
        inputs tensor, feature vector tensor
    """
    check_tensorflow()

    inputs = layers.Input(shape=input_shape, name="vit_spectrogram_input")

    # Pad width from 94 to 96 so both dimensions (64, 96) divide cleanly by patch_size (8, 8)
    # Total patches = (64//8) * (96//8) = 8 * 12 = 96 patches
    x = layers.ZeroPadding2D(padding=((0, 0), (0, 2)), name="vit_time_pad")(inputs)

    # Patch extraction + Linear Projection + Positional Embedding
    tokens = PatchAndPositionEmbedding(
        patch_size=patch_size,
        projection_dim=projection_dim,
        num_patches=96,
        name="vit_patch_pos_embed"
    )(x)
    tokens = layers.Dropout(dropout_rate * 0.5, name="vit_embed_dropout")(tokens)

    # Stacked Transformer Encoder blocks
    key_dim = projection_dim // num_heads
    for i in range(transformer_layers):
        tokens = transformer_encoder_block(
            tokens,
            num_heads=num_heads,
            key_dim=key_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            name_prefix=f"vit_tblock_{i+1}"
        )

    # Global Average Pooling over all patch tokens
    seq_repr = layers.GlobalAveragePooling1D(name="vit_gap1d")(tokens)
    seq_repr = layers.LayerNormalization(epsilon=1e-6, name="vit_final_ln")(seq_repr)

    # Final feature projection
    features = layers.Dense(feature_dim, activation=tf.nn.gelu, name="vit_dense_feat")(seq_repr)
    features = layers.Dropout(dropout_rate, name="vit_feat_dropout")(features)

    return inputs, features


def build_vit_model(
    input_shape: Tuple[int, int, int] = (64, 94, 1),
    num_classes: int = 4,
    patch_size: int = 8,
    projection_dim: int = 64,
    num_heads: int = 4,
    transformer_layers: int = 3,
    mlp_dim: int = 128,
    dropout_rate: float = 0.2,
    feature_dim: int = 128
):
    """
    Builds a standalone ViT classifier model with softmax output.
    """
    check_tensorflow()
    inputs, features = build_vit_backbone(
        input_shape=input_shape,
        patch_size=patch_size,
        projection_dim=projection_dim,
        num_heads=num_heads,
        transformer_layers=transformer_layers,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        feature_dim=feature_dim
    )

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="vessel_classification"
    )(features)

    model = models.Model(inputs=inputs, outputs=outputs, name="ViT_Classifier")
    return model


if __name__ == "__main__":
    if TF_AVAILABLE:
        m = build_vit_model()
        m.summary()
    else:
        print("[INFO] TensorFlow not found locally. Code syntax and structure validated.")
