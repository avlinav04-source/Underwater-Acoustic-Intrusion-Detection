"""
evaluate.py
-----------
Comprehensive model evaluation on the independent, leak-free test split.

Calculates and reports:
- Test Loss and Test Accuracy
- Per-Class Precision, Recall, and F1-Scores
- Macro and Weighted F1-Scores
- Confusion Matrix and Normalized Confusion Matrix Heatmaps
- Detailed Classification Report
- Academic documentation of dataset support and Tug minority limitations.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config,
    ensure_dir,
    save_json,
    load_json,
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    tf = None
    load_model = None
    TF_AVAILABLE = False


def plot_confusion_matrices(
    cm: np.ndarray,
    classes: list,
    output_dir: Path,
    model_name: str = "Hybrid_Model"
):
    """Plots and saves standard and normalized confusion matrix heatmaps."""
    ensure_dir(output_dir)

    # 1. Raw counts
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix (Counts) - {model_name}",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )
    plt.tight_layout()
    raw_cm_path = output_dir / f"confusion_matrix_{model_name.lower()}.png"
    plt.savefig(raw_cm_path, dpi=150)
    plt.close(fig)

    # 2. Row-normalized (Recall)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, format="%.2f")
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Normalized Confusion Matrix - {model_name}",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontweight="bold"
            )
    plt.tight_layout()
    norm_cm_path = output_dir / f"confusion_matrix_normalized_{model_name.lower()}.png"
    plt.savefig(norm_cm_path, dpi=150)
    plt.close(fig)

    print(f"[+] Confusion matrices saved to: {output_dir.resolve()}")


def evaluate_model(
    model_path: Optional[str] = None,
    test_features_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates trained model on the unseen test set.
    """
    if not TF_AVAILABLE:
        print("\n" + "!" * 70)
        print(" [ERROR] TensorFlow is required to load and evaluate .keras models.")
        print(" Please evaluate in Google Colab or install tensorflow locally.")
        print("!" * 70 + "\n")
        return {}

    config = load_config()
    paths_cfg = config.get("paths", {})

    models_dir = PROJECT_ROOT / paths_cfg.get("models_dir", "models")
    target_model_path = Path(model_path or (models_dir / "best_hybrid_model.keras"))
    target_out_dir = Path(output_dir or (PROJECT_ROOT / paths_cfg.get("outputs_dir", "outputs")))
    features_dir = PROJECT_ROOT / paths_cfg.get("features_dir", "outputs/features")
    test_npz = Path(test_features_path or (features_dir / "test_features.npz"))

    if not target_model_path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {target_model_path.resolve()}.\n"
            f"Please run 'src/train.py' first to produce a real trained model."
        )

    if not test_npz.is_file():
        raise FileNotFoundError(
            f"Test features file not found at: {test_npz.resolve()}.\n"
            f"Please run 'src/train.py' or feature extraction first."
        )

    print(f"\n[+] Loading trained model from: {target_model_path.name}")
    # Custom objects handling for ViT PatchAndPositionEmbedding
    from src.model_vit import PatchAndPositionEmbedding
    custom_objects = {"PatchAndPositionEmbedding": PatchAndPositionEmbedding}
    model = load_model(str(target_model_path), custom_objects=custom_objects)

    print(f"[+] Loading test features from: {test_npz.name}")
    test_data = np.load(str(test_npz))
    X_test = test_data["X"]
    y_test = test_data["y"]

    print(f"[+] Running inference on {len(X_test)} unseen test segments...")
    y_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    # Detailed per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=list(range(len(CANONICAL_CLASSES))), zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CANONICAL_CLASSES))))

    report_dict = classification_report(
        y_test, y_pred,
        target_names=CANONICAL_CLASSES,
        output_dict=True,
        zero_division=0
    )

    per_class_summary = {}
    for idx, cls_name in enumerate(CANONICAL_CLASSES):
        per_class_summary[cls_name] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1_score": round(float(f1_class[idx]), 4),
            "test_segment_support": int(support_class[idx]),
        }

    results = {
        "model_checkpoint": target_model_path.name,
        "test_total_segments": int(len(X_test)),
        "metrics": {
            "test_accuracy": round(acc, 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weight), 4),
        },
        "per_class_metrics": per_class_summary,
        "confusion_matrix": cm.tolist(),
        "academic_limitations_note": (
            "Evaluation performed strictly on recordings isolated at the recording level. "
            "Tug evaluation is based on segments from the single held-out test recording due to "
            "the severe dataset constraint (only 3 total Tug recordings in DeepShip)."
        )
    }

    # Save JSON report
    report_json_path = target_out_dir / "evaluation_report.json"
    save_json(results, report_json_path)

    # Plot confusion matrix
    plot_confusion_matrices(cm, CANONICAL_CLASSES, target_out_dir, model_name=target_model_path.stem)

    # Print summary to console
    print("\n" + "=" * 70)
    print(" INDEPENDENT TEST SET EVALUATION REPORT")
    print("=" * 70)
    print(f"Model Checkpoint : {target_model_path.name}")
    print(f"Total Test Segs  : {len(X_test)}")
    print(f"Test Accuracy    : {acc * 100:.2f}%")
    print(f"Macro F1-Score   : {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weight:.4f}")
    print("-" * 70)
    print(f"{'Class':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
    print("-" * 70)
    for cls_name, m in per_class_summary.items():
        print(
            f"{cls_name:<16} | "
            f"{m['precision']:<10.4f} | "
            f"{m['recall']:<10.4f} | "
            f"{m['f1_score']:<10.4f} | "
            f"{m['test_segment_support']}"
        )
    print("=" * 70)
    print(f"[+] Evaluation JSON report saved to: {report_json_path.resolve()}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on unseen test set.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to .keras model checkpoint")
    parser.add_argument("--test_npz", type=str, default=None, help="Path to test_features.npz")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation reports")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_npz, args.output_dir)
