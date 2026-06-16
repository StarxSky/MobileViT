import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


COCO_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (192, 192, 192), (255, 165, 0), (255, 20, 147),
]

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_color_for_label(label):
    return COCO_COLORS[label % len(COCO_COLORS)]


def get_category_name(label):
    if 1 <= label <= 80:
        return COCO_CATEGORIES[label - 1]
    return f"cls_{label}"


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: List,
    labels: List,
    scores: Optional[List] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_label: bool = True,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes on image. boxes in [x1, y1, x2, y2] pixel coords."""
    img = image.copy()
    h, w = img.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        clr = get_color_for_label(labels[i]) if color is None else color
        cv2.rectangle(img, (x1, y1), (x2, y2), clr, thickness)

        if show_label:
            label_str = get_category_name(labels[i])
            if scores is not None:
                label_str = f"{label_str} {scores[i]:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            y1_text = max(y1 - text_h - baseline, 0)
            cv2.rectangle(
                img, (x1, y1_text), (x1 + text_w, y1_text + text_h + baseline),
                clr, -1,
            )
            cv2.putText(
                img, label_str, (x1, y1_text + text_h),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
            )

    return img


def visualize_predictions(
    image: np.ndarray,
    gt_boxes: List,
    gt_labels: List,
    pred_boxes: List,
    pred_labels: List,
    pred_scores: List,
    save_path: str,
    image_size: int = 320,
):
    """Side-by-side comparison: GT vs predictions."""
    if not _HAS_MPL:
        _fallback_pil_comparison(
            image, gt_boxes, gt_labels, pred_boxes, pred_labels,
            pred_scores, save_path, image_size,
        )
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # GT
    gt_img = draw_boxes_on_image(
        image, gt_boxes, gt_labels, color=None, show_label=True
    )
    ax1.imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Ground Truth ({len(gt_boxes)} objects)")
    ax1.axis("off")

    # Predictions
    pred_img = draw_boxes_on_image(
        image, pred_boxes, pred_labels, pred_scores, color=None, show_label=True
    )
    ax2.imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Predictions ({len(pred_boxes)} detections)")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fallback_pil_comparison(
    image, gt_boxes, gt_labels, pred_boxes, pred_labels,
    pred_scores, save_path, image_size,
):
    """Fallback when matplotlib not available."""
    gt_img = draw_boxes_on_image(image, gt_boxes, gt_labels)
    pred_img = draw_boxes_on_image(image, pred_boxes, pred_labels, pred_scores)

    # Stack side by side
    h, w = image.shape[:2]
    canvas = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
    canvas[:, :w] = gt_img
    canvas[:, w + 10:] = pred_img
    cv2.imwrite(save_path, canvas[..., ::-1])  # BGR to RGB


def visualize_batch(
    images: torch.Tensor,
    targets: List[Dict],
    predictions: Optional[Dict] = None,
    save_dir: str = "./visualizations",
    prefix: str = "batch",
    max_samples: int = 8,
    denormalize: bool = True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Visualize a batch of images with GT (and optional predictions)."""
    os.makedirs(save_dir, exist_ok=True)
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)

    n = min(images.size(0), max_samples)
    for b in range(n):
        img = images[b].cpu().numpy().transpose(1, 2, 0)
        if denormalize:
            img = img * std + mean
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

        img_h, img_w = img.shape[:2]

        # GT boxes (cxcywh -> xyxy, denormalize)
        gt_boxes = targets[b]["boxes"].cpu()
        gt_labels = targets[b]["labels"].cpu().tolist()
        gt_boxes_xyxy = []
        for box in gt_boxes:
            cx, cy, w, h = box.tolist()
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            gt_boxes_xyxy.append([x1, y1, x2, y2])

        vis_img = draw_boxes_on_image(
            img, gt_boxes_xyxy, gt_labels, color=None, show_label=True
        )

        # Predictions
        if predictions is not None:
            pred_boxes = predictions["boxes"][b].cpu()
            pred_scores = predictions["scores"][b].cpu()
            pred_labels = predictions["labels"][b].cpu().tolist()
            pred_boxes_xyxy = []
            for box in pred_boxes:
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                pred_boxes_xyxy.append([x1, y1, x2, y2])

            vis_img = draw_boxes_on_image(
                vis_img, pred_boxes_xyxy, pred_labels, pred_scores,
                color=(255, 0, 0), show_label=True,
            )

        save_path = os.path.join(save_dir, f"{prefix}_{b}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


def plot_metrics_history(
    history: List[Dict],
    save_path: str,
    metrics: Optional[List[str]] = None,
    title: str = "Training Metrics",
):
    """Plot metric curves over epochs."""
    if not _HAS_MPL:
        return

    if metrics is None:
        metrics = ["AP", "AP50", "AP_small", "AP_medium", "AP_large",
                    "AP_tiny", "AP_extreme_tiny", "AP75_small"]

    keys = [m for m in metrics if any(m in e for e in history)]
    if not keys:
        return

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14)

    for ax, key in zip(axes, keys):
        epochs = []
        vals = []
        for entry in history:
            if key in entry:
                epochs.append(entry["epoch"])
                vals.append(entry[key])
        if not epochs:
            continue
        ax.plot(epochs, vals, "-o", markersize=3, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)

        # Annotate best value
        best_idx = np.argmax(vals)
        ax.annotate(
            f"{vals[best_idx]:.3f}",
            xy=(epochs[best_idx], vals[best_idx]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="green",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_analysis(
    error_metrics: Dict,
    save_path: str,
    title: str = "Detection Error Analysis",
):
    """Bar chart of error types (FP analysis)."""
    if not _HAS_MPL:
        return

    labels = {
        "AP_C75": "AP@IoU=0.75",
        "AP_C50": "AP@IoU=0.50",
        "AP_C25": "AP@IoU=0.25",
        "loc_error": "Loc Error (C25→C75)",
        "bg_confusion": "BG Confusion (C25→C50)",
    }

    keys = [k for k in labels if k in error_metrics]
    if not keys:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # AP at different IoUs (bar chart)
    ap_keys = [k for k in keys if k.startswith("AP_")]
    ap_vals = [error_metrics[k] for k in ap_keys]
    ax1.bar(range(len(ap_keys)), ap_vals, color=["#2ecc71", "#3498db", "#9b59b6"])
    ax1.set_xticks(range(len(ap_keys)))
    ax1.set_xticklabels([labels[k] for k in ap_keys], rotation=15)
    ax1.set_ylabel("AP")
    ax1.set_title("Precision at IoU Thresholds")
    ax1.grid(True, axis="y", alpha=0.3)

    # Error breakdown (bar chart)
    err_keys = [k for k in keys if k.startswith(("loc", "bg"))]
    err_vals = [error_metrics[k] * 100 for k in err_keys]  # as percentage
    colors = ["#e74c3c", "#f39c12"]
    ax2.bar(range(len(err_keys)), err_vals, color=colors[:len(err_keys)])
    ax2.set_xticks(range(len(err_keys)))
    ax2.set_xticklabels([labels[k] for k in err_keys], rotation=15)
    ax2.set_ylabel("Drop (%)")
    ax2.set_title("Error Sources")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_small_object_analysis(
    small_metrics: Dict,
    save_path: str,
    title: str = "Small Object Analysis",
):
    """Visualize small object detection metrics."""
    if not _HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Detection rate by size
    if "detection_rate" in small_metrics:
        dr = small_metrics["detection_rate"]
        sizes = list(dr.keys())
        rates = [dr[s] * 100 for s in sizes]
        colors_bar = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
        axes[0].bar(sizes, rates, color=colors_bar[:len(sizes)])
        axes[0].set_ylabel("Detection Rate (%)")
        axes[0].set_title("Detection Rate by Size")
        axes[0].grid(True, axis="y", alpha=0.3)
        for i, (s, r) in enumerate(zip(sizes, rates)):
            axes[0].text(i, r + 1, f"{r:.1f}%", ha="center", fontsize=8)

    # 2. GT vs detection count by size
    if "gt_distribution" in small_metrics and "det_distribution" in small_metrics:
        gt = small_metrics["gt_distribution"]
        det = small_metrics["det_distribution"]
        sizes = [k for k in ["tiny", "small", "medium", "large"] if k in gt]
        x = np.arange(len(sizes))
        w = 0.35
        axes[1].bar(x - w / 2, [gt[s] for s in sizes], w, label="GT", color="#95a5a6")
        axes[1].bar(x + w / 2, [det[s] for s in sizes], w, label="Detections", color="#2ecc71")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(sizes)
        axes[1].set_ylabel("Count")
        axes[1].set_title("GT vs Detections by Size")
        axes[1].legend()
        axes[1].grid(True, axis="y", alpha=0.3)

    # 3. Recall for tiny/small objects
    recall_keys = [k for k in ["AR_tiny", "AR_extreme", "AR_small", "AR_medium", "AR_large"]
                   if k in small_metrics]
    recall_vals = [small_metrics[k] * 100 for k in recall_keys]
    colors_r = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(recall_keys)))
    axes[2].barh(recall_keys, recall_vals, color=colors_r)
    axes[2].set_xlabel("AR@100 (%)")
    axes[2].set_title("Average Recall by Size")
    for i, (k, v) in enumerate(zip(recall_keys, recall_vals)):
        axes[2].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_feature_maps(
    features: Dict[str, torch.Tensor],
    save_dir: str,
    prefix: str = "features",
    num_channels: int = 8,
):
    """Visualize feature maps from the backbone/SSD heads."""
    os.makedirs(save_dir, exist_ok=True)
    if not _HAS_MPL:
        return

    for name, fm in features.items():
        if fm.dim() != 4:
            continue
        B, C, H, W = fm.shape
        sample = fm[0:1]  # first batch
        n_plot = min(C, num_channels)
        fig, axes = plt.subplots(1, n_plot, figsize=(n_plot * 2, 2))
        if n_plot == 1:
            axes = [axes]

        # Normalize each channel to [0,1]
        for i in range(n_plot):
            ch = sample[0, i].detach().cpu().numpy()
            ch = (ch - ch.min()) / max(ch.max() - ch.min(), 1e-8)
            axes[i].imshow(ch, cmap="viridis")
            axes[i].axis("off")
            axes[i].set_title(f"ch{i}")

        fig.suptitle(f"{name} ({H}x{W})")
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f"{prefix}_{name}.jpg"),
            dpi=120, bbox_inches="tight",
        )
        plt.close(fig)


def make_visualization_grid(
    image_dir: str,
    output_path: str,
    n_cols: int = 4,
    titles: Optional[List[str]] = None,
):
    """Combine multiple visualization images into a grid."""
    if not _HAS_MPL:
        return

    import glob

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        return

    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, img_path in enumerate(images):
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
