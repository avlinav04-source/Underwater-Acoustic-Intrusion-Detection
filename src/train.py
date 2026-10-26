"""
train.py
--------
Reproducible training pipeline for Underwater Acoustic Intrusion Detection.

Supports training the Hybrid DWS-CNN + ViT architecture, standalone DWS-CNN,
or standalone ViT. Incorporates class-weighted loss balancing, learning rate
scheduling, early stopping, best model checkpointing, history logging, and
training visualization curves.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config,
    ensure_dir,
    save_json,
    load_json,
    set_seed,
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)
from src.feature_extraction import extract_features_for_split

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, optimizers, losses
    TF_AVAILABLE = True
except ImportError:
    tf = None
    callbacks = None
    optimizers = None
    losses = None
    TF_AVAILABLE = False


def load_or_extract_split_features(
    manifest_path: Path,
    config: Dict[str, Any],
    features_dir: Path,
    force_recompute: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[int, float]]:
    """
    Loads pre-extracted features from disk or extracts them from raw audio.
    Saves cached .npz files in outputs/features/ for fast iteration.
    """
    ensure_dir(features_dir)
    manifest = load_json(manifest_path)
    raw_data_dir = config["dataset"]["raw_data_dir"]

    splits_data = {}
    for split_name in ["train", "val", "test"]:
        cache_file = features_dir / f"{split_name}_features.npz"

        if cache_file.is_file() and not force_recompute:
            print(f"[+] Loading cached {split_name} features from: {cache_file.name}")
            data = np.load(str(cache_file))
            X = data["X"]
            y = data["y"]
        else:
            print(f"[+] Extracting Log-Mel features for '{split_name}' split ({len(manifest['splits'][split_name])} recordings)...")
            X, y, rec_ids = extract_features_for_split(
                records=manifest["splits"][split_name],
                raw_data_dir=raw_data_dir,
                config=config
            )
            np.savez_compressed(str(cache_file), X=X, y=y, rec_ids=np.array(rec_ids))
            print(f"    Saved {len(X)} segments to {cache_file.name}")

        splits_data[split_name] = (X, y)

    # Class weights from split manifest
    raw_weights = manifest.get("class_weights", {})
    class_weights = {int(k): float(v) for k, v in raw_weights.items()}

    return (
        splits_data["train"],
        splits_data["val"],
        splits_data["test"],
        class_weights
    )


def plot_training_curves(history_df: pd.DataFrame, output_path: Path):
    """Plots and saves loss and accuracy learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history_df["epoch"], history_df["loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history_df.columns:
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Accuracy curve
    axes[1].plot(history_df["epoch"], history_df["accuracy"], label="Train Accuracy", color="#2ca02c", linewidth=2)
    if "val_accuracy" in history_df.columns:
        axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Val Accuracy", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[+] Training curves saved to: {output_path.resolve()}")


def train_model(
    model_type: str = "hybrid",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    force_recompute_features: bool = False,
    quick_test: bool = False
):
    """
    Main training execution function.
    """
    if not TF_AVAILABLE:
        print("\n" + "!" * 70)
        print(" [ERROR] TensorFlow is not installed in the current environment.")
        print(" To execute model training, either:")
        print(" 1. Run in Google Colab / Kaggle with GPU acceleration (recommended).")
        print("    See notebooks/training_guide.md for the ready-to-run workflow.")
        print(" 2. Install TensorFlow in this environment:")
        print("    pip install tensorflow")
        print("!" * 70 + "\n")
        return None

    config = load_config()
    train_cfg = config.get("training", {})
    split_cfg = config.get("split", {})
    paths_cfg = config.get("paths", {})

    seed = split_cfg.get("random_seed", 42)
    set_seed(seed)

    num_epochs = epochs or (2 if quick_test else train_cfg.get("epochs", 35))
    bs = batch_size or train_cfg.get("batch_size", 16)
    lr = learning_rate or train_cfg.get("learning_rate", 0.0005)

    manifest_path = PROJECT_ROOT / paths_cfg.get("split_manifest", "outputs/dataset_split.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at: {manifest_path}. "
            "Please run 'src/dataset_split.py' first."
        )

    features_dir = PROJECT_ROOT / paths_cfg.get("features_dir", "outputs/features")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights = load_or_extract_split_features(
        manifest_path=manifest_path,
        config=config,
        features_dir=features_dir,
        force_recompute=force_recompute_features
    )

    print(f"\n[+] Dataset Segments Extracted:")
    print(f"    * Train : {len(X_train)} segments")
    print(f"    * Val   : {len(X_val)} segments")
    print(f"    * Test  : {len(X_test)} segments")

    if quick_test:
        X_train, y_train = X_train[:32], y_train[:32]
        X_val, y_val = X_val[:16], y_val[:16]

    input_shape = X_train.shape[1:]  # (64, 94, 1)
    num_classes = len(CANONICAL_CLASSES)

    # Instantiate chosen architecture
    if model_type == "hybrid":
        from src.model_hybrid import build_hybrid_model
        model = build_hybrid_model(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "dws_cnn":
        from src.model_dws_cnn import build_dws_cnn_model
        model = build_dws_cnn_model(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "vit":
        from src.model_vit import build_vit_model
        model = build_vit_model(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'hybrid', 'dws_cnn', or 'vit'.")

    model.summary()

    # Compile with Adam and SparseCategoricalCrossentropy
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_fn = losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # Setup directories and callbacks
    models_dir = ensure_dir(PROJECT_ROOT / paths_cfg.get("models_dir", "models"))
    outputs_dir = ensure_dir(PROJECT_ROOT / paths_cfg.get("outputs_dir", "outputs"))

    checkpoint_path = models_dir / f"best_{model_type}_model.keras"
    history_csv_path = outputs_dir / f"training_history_{model_type}.csv"
    curves_png_path = outputs_dir / f"training_curves_{model_type}.png"

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg.get("early_stopping_patience", 8),
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg.get("reduce_lr_factor", 0.5),
            patience=train_cfg.get("reduce_lr_patience", 4),
            min_lr=train_cfg.get("min_lr", 0.00001),
            verbose=1
        ),
        callbacks.CSVLogger(str(history_csv_path))
    ]

    print(f"\n[+] Starting training for {num_epochs} epochs (batch size = {bs}, lr = {lr})...")
    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=bs,
        class_weight=class_weights if train_cfg.get("use_class_weights", True) else None,
        callbacks=cb_list,
        verbose=1
    )

    elapsed_time = time.time() - start_time
    print(f"\n[+] Training completed in {elapsed_time:.1f} seconds ({elapsed_time/60.0:.2f} minutes).")

    # Save model metadata
    metadata = {
        "model_type": model_type,
        "input_shape": list(input_shape),
        "classes": CANONICAL_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "audio_config": config.get("audio", {}),
        "features_config": config.get("features", {}),
        "training_epochs_completed": len(history.history["loss"]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]) if "val_accuracy" in history.history else None,
        "checkpoint_file": checkpoint_path.name,
    }
    save_json(metadata, models_dir / "model_metadata.json")

    # Plot curves
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = range(1, len(history_df) + 1)
    plot_training_curves(history_df, curves_png_path)

    return model, history_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train underwater acoustic classification model.")
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["hybrid", "dws_cnn", "vit"], help="Model branch to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--recompute_features", action="store_true", help="Force recomputation of audio features")
    parser.add_argument("--quick_test", action="store_true", help="Run 2 quick verification epochs on sample subset")
    args = parser.parse_args()

    train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        force_recompute_features=args.recompute_features,
        quick_test=args.quick_test
    )
