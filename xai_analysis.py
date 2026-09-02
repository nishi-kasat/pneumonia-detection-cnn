"""End-to-end explainability analysis for the pneumonia CNN.

Usage (after training the original model):
    python xai_analysis.py --model pneumonia_cnn_model.h5 \
        --data-dir /content/chest_xray --output-dir outputs/xai

The script evaluates the classifier, selects representative TP/TN/FP/FN
examples, generates Grad-CAM explanations, and measures explanation
stability under small input perturbations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from explainability.gradcam import make_gradcam_heatmap, overlay_gradcam


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the saved Keras model")
    parser.add_argument("--data-dir", required=True, help="Path containing train/test folders")
    parser.add_argument("--output-dir", default="outputs/xai")
    parser.add_argument("--img-size", type=int, default=150)
    parser.add_argument("--max-per-class", type=int, default=250)
    return parser.parse_args()


def load_test_data(data_dir: Path, img_size: int, max_per_class: int):
    """Load a deterministic, bounded test set for reproducible XAI analysis."""
    images, labels, paths = [], [], []
    test_dir = data_dir / "test"

    for label, class_name in enumerate(CLASS_NAMES):
        files = sorted(test_dir.joinpath(class_name).glob("*.jpeg"))
        files += sorted(test_dir.joinpath(class_name).glob("*.jpg"))
        files = files[:max_per_class]
        for path in files:
            img = load_img(path, target_size=(img_size, img_size), color_mode="rgb")
            arr = img_to_array(img).astype("float32") / 255.0
            images.append(arr)
            labels.append(label)
            paths.append(path)

    if not images:
        raise FileNotFoundError(f"No test images found under {test_dir}")
    return np.stack(images), np.asarray(labels), paths


def evaluate(model, x, y, output_dir):
    scores = model.predict(x, verbose=0).reshape(-1)
    preds = (scores >= 0.5).astype(int)

    print(classification_report(y, preds, target_names=CLASS_NAMES, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y, scores):.4f}")

    cm = confusion_matrix(y, preds)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot()
    plt.title("Pneumonia CNN — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    return scores, preds


def representative_indices(y, preds):
    cases = {
        "true_positive": np.where((y == 1) & (preds == 1))[0],
        "true_negative": np.where((y == 0) & (preds == 0))[0],
        "false_positive": np.where((y == 0) & (preds == 1))[0],
        "false_negative": np.where((y == 1) & (preds == 0))[0],
    }
    return {name: int(indices[0]) for name, indices in cases.items() if len(indices)}


def save_explanation(model, x, y, scores, idx, name, paths, output_dir):
    sample = x[idx : idx + 1]
    heatmap, pneumonia_score, layer_name = make_gradcam_heatmap(sample, model)
    base = np.uint8(np.clip(x[idx] * 255.0, 0, 255))
    overlay = overlay_gradcam(base, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(base)
    axes[0].set_title(f"Original\nTrue: {CLASS_NAMES[y[idx]]}")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Pred: {CLASS_NAMES[int(scores[idx] >= 0.5)]}\n"
        f"P(pneumonia)={scores[idx]:.3f}"
    )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{name.replace('_', ' ').title()} — {paths[idx].name}\nLayer: {layer_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return heatmap


def heatmap_similarity(a, b):
    """Cosine similarity between two normalized Grad-CAM maps."""
    a = cv2.resize(a, (64, 64)).reshape(-1).astype("float32")
    b = cv2.resize(b, (64, 64)).reshape(-1).astype("float32")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def stability_check(model, x, idx):
    """Measure explanation similarity after a small brightness perturbation."""
    original = x[idx : idx + 1]
    perturbed = np.clip(original * 1.05, 0.0, 1.0)
    heat_a, _, _ = make_gradcam_heatmap(original, model)
    heat_b, _, _ = make_gradcam_heatmap(perturbed, model)
    return heatmap_similarity(heat_a, heat_b)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(args.model)
    print(f"Model output: {model.output_shape}")

    x, y, paths = load_test_data(Path(args.data_dir), args.img_size, args.max_per_class)
    print(f"Loaded {len(x)} test images")

    scores, preds = evaluate(model, x, y, output_dir)
    representatives = representative_indices(y, preds)

    stability = {}
    for name, idx in representatives.items():
        save_explanation(model, x, y, scores, idx, name, paths, output_dir)
        stability[name] = stability_check(model, x, idx)

    print("\nRepresentative explanation stability (brightness +5%):")
    for name, similarity in stability.items():
        print(f"  {name:16s}: cosine similarity = {similarity:.4f}")

    with open(output_dir / "stability.txt", "w", encoding="utf-8") as f:
        for name, similarity in stability.items():
            f.write(f"{name}\t{similarity:.6f}\n")

    print(f"\nSaved XAI outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
