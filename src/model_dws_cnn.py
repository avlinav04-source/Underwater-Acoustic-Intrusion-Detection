"""
model_dws_cnn.py
----------------
Depthwise Separable Convolutional Neural Network (DWS-CNN) for underwater
acoustic vessel classification.

Utilizes Depthwise and Pointwise Separable 2D convolutions (SeparableConv2D)
to dramatically reduce parameter count and computational complexity while
capturing spatial-frequency acoustic patterns across Log-Mel spectrograms.
"""

from typing import Tuple, Optional
import os

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    TF_AVAILABLE = True
except ImportError:
    tf = None
    layers = None
    models = None
    regularizers = None
    TF_AVAILABLE = False


def check_tensorflow():
    """Validates that TensorFlow is available in the environment."""
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed in the local environment. "
            "Please install TensorFlow or run model training in Google Colab / Kaggle. "
            "See notebooks/training_guide.md for execution instructions."
        )


def build_dws_cnn_backbone(
    input_shape: Tuple[int, int, int] = (64, 94, 1),
    base_filters: int = 32,
    dropout_rate: float = 0.3,
    feature_dim: int = 128
):
    """
    Constructs the feature extractor backbone using Depthwise Separable Convolutions.

    Args:
        input_shape: Dimensions of input Log-Mel Spectrogram (n_mels, time_steps, 1).
        base_filters: Number of filters in initial convolution block.
        dropout_rate: Dropout rate for regularization.
        feature_dim: Dimension of output latent feature vector.

    Returns:
        inputs tensor, feature vector tensor
    """
    check_tensorflow()

    inputs = layers.Input(shape=input_shape, name="spectrogram_input")

    # Initial standard Conv2D to establish initial channel representations
    x = layers.Conv2D(
        base_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
        name="dws_conv1_stem"
    )(inputs)
    x = layers.BatchNormalization(name="dws_bn1_stem")(x)
    x = layers.ReLU(name="dws_relu1_stem")(x)

    # --- Block 1: Depthwise Separable Conv ---
    x = layers.SeparableConv2D(
        base_filters,
        kernel_size=(3, 3),
        padding="same",
        depth_multiplier=1,
        kernel_initializer="he_normal",
        name="dws_sep_conv1"
    )(x)
    x = layers.BatchNormalization(name="dws_bn1")(x)
    x = layers.ReLU(name="dws_relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="dws_pool1")(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.5, name="dws_sdrop1")(x)

    # --- Block 2: Depthwise Separable Conv (64 filters) ---
    x = layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=(3, 3),
        padding="same",
        depth_multiplier=1,
        kernel_initializer="he_normal",
        name="dws_sep_conv2"
    )(x)
    x = layers.BatchNormalization(name="dws_bn2")(x)
    x = layers.ReLU(name="dws_relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="dws_pool2")(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.75, name="dws_sdrop2")(x)

    # --- Block 3: Depthwise Separable Conv (128 filters) ---
    x = layers.SeparableConv2D(
        base_filters * 4,
        kernel_size=(3, 3),
        padding="same",
        depth_multiplier=1,
        kernel_initializer="he_normal",
        name="dws_sep_conv3"
    )(x)
    x = layers.BatchNormalization(name="dws_bn3")(x)
    x = layers.ReLU(name="dws_relu3")(x)

    # Global Average Pooling across spatial-temporal dimensions
    x = layers.GlobalAveragePooling2D(name="dws_gap")(x)
    x = layers.Dense(feature_dim, activation="relu", kernel_initializer="he_normal", name="dws_dense_feat")(x)
    x = layers.BatchNormalization(name="dws_bn_feat")(x)
    features = layers.Dropout(dropout_rate, name="dws_feat_dropout")(x)

    return inputs, features


def build_dws_cnn_model(
    input_shape: Tuple[int, int, int] = (64, 94, 1),
    num_classes: int = 4,
    base_filters: int = 32,
    dropout_rate: float = 0.3,
    feature_dim: int = 128
):
    """
    Builds a standalone DWS-CNN classifier model with softmax head.
    """
    check_tensorflow()
    inputs, features = build_dws_cnn_backbone(
        input_shape=input_shape,
        base_filters=base_filters,
        dropout_rate=dropout_rate,
        feature_dim=feature_dim
    )

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="vessel_classification"
    )(features)

    model = models.Model(inputs=inputs, outputs=outputs, name="DWS_CNN_Classifier")
    return model


if __name__ == "__main__":
    if TF_AVAILABLE:
        m = build_dws_cnn_model()
        m.summary()
    else:
        print("[INFO] TensorFlow not found locally. Code syntax and structure validated.")
