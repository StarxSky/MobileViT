#!/usr/bin/env python3
"""
Ablation Study: Ghost Conv MobileViT-S SSD Inference on COCO val2017.

Compares Ghost Conv variant against standard MobileViT-S SSD and Apple's
official ml-cvnets results for ablation experiments.

Usage:
  # Evaluate a trained Ghost model
  python inference_coco.py --checkpoint outputs/model_best.pth --data-dir ./coco

  # Evaluate standard model (no Ghost Conv)
  python inference_coco.py --checkpoint outputs/model_best.pth --data-dir ./coco --no-use-ghost

  # Load Apple pretrained weights directly (baseline)
  python inference_coco.py --apple-weights apple_ssd_mobilevit.pt --data-dir ./coco

  # Compare both variants side by side (ablation)
  python inference_coco.py --checkpoint ghost_model.pth --checkpoint-std standard_model.pth --data-dir ./coco

  # Use EMA weights
  python inference_coco.py --checkpoint outputs/model_best.pth --data-dir ./coco --no-ema

Reference:
  Apple ml-cvnets official: https://github.com/apple/ml-cvnets
  Paper: MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (ICLR 2022)
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms

try:
    from pycocotools.coco import COCO
    _HAS_COCO = True
except ImportError:
    _HAS_COCO = False

from ghost_mobilevit_ssd.ssd import GhostMobileViTSSD
from ghost_mobilevit_ssd.utils import load_apple_pretrained_weights
from ghost_mobilevit_ssd.metrics import (
    COCOEvaluator, SmallObjectAnalyzer, DetectionErrorAnalyzer,
)


APPLE_OFFICIAL_METRICS = {
    "AP": 22.3,
    "AP50": 38.2,
    "AP75": 22.0,
    "AP_small": 7.0,
    "AP_medium": 22.5,
    "AP_large": 36.8,
    "AR1": 18.5,
    "AR10": 30.8,
    "AR100": 33.0,
    "AR_small": 11.0,
    "AR_medium": 35.4,
    "AR_large": 51.8,
}

APPLE_REFERENCE_DESC = """
Reference: Apple ml-cvnets official results (MobileViT-S + SSDLite 320)
  Paper: MobileViT: Light-weight, General-purpose, and Mobile-friendly
         Vision Transformer, Mehta & Rastegari, ICLR 2022
  Code:  https://github.com/apple/ml-cvnets
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation: Ghost Conv MobileViT-S SSD Inference on COCO"
    )
    parser.add_argument("--data-dir", type=str, default="./coco",
                        help="COCO dataset root")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--checkpoint-std", type=str, default=None,
                        help="Path to standard (no Ghost) checkpoint for comparison")
    parser.add_argument("--apple-weights", type=str, default=None,
                        help="Load Apple ml-cvnets pretrained weights")
    parser.add_argument("--use-ghost", action="store_true", default=None,
                        help="Use Ghost convolutions")
    parser.add_argument("--no-use-ghost", action="store_true", default=None,
                        dest="no_ghost")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA weights from checkpoint")
    parser.add_argument("--no-ema", action="store_false", dest="use_ema")
    parser.add_argument("--ghost-ratio", type=float, default=0.5,
                        help="Ghost ratio")
    parser.add_argument("--image-size", type=int, default=320,
                        help="Image size")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loading workers")
    parser.add_argument("--output-dir", type=str, default="./ablation_results",
                        help="Output directory")
    parser.add_argument("--n-classes", type=int, default=91,
                        help="Number of classes (including background)")
    parser.add_argument("--conf-threshold", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--objects-per-image", type=int, default=200)
    parser.add_argument("--use-fpn", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Run on a subset only")
    parser.add_argument("--dry-run-samples", type=int, default=100)
    return parser.parse_args()


def build_model(n_classes, use_ghost, ghost_ratio, image_size, use_fpn=False):
    return GhostMobileViTSSD(
        n_classes=n_classes,
        ghost_ratio=ghost_ratio,
        use_ghost=use_ghost,
        image_size=(image_size, image_size),
        use_fpn=use_fpn,
    )


def load_checkpoint(model, checkpoint_path, use_ema=True):
    print(f"    Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "model_state_dict" in state:
        if use_ema and "ema_state_dict" in state and state["ema_state_dict"] is not None:
            model.load_state_dict(state["ema_state_dict"], strict=False)
            src = "EMA"
        else:
            model.load_state_dict(state["model_state_dict"], strict=False)
            src = "model"
        epoch = state.get("epoch", -1)
        best_map = state.get("best_map", 0.0)
        print(f"      Loaded {src} weights | epoch={epoch} | best_mAP={best_map:.4f}")
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
        print(f"      Loaded from state_dict")
    else:
        model.load_state_dict(state, strict=False)
        print(f"      Loaded raw state_dict")
    return model


def build_label_to_catid(coco):
    cat_ids = coco.getCatIds()
    return {i + 1: cid for i, cid in enumerate(cat_ids)}


class COCOValDataset(Dataset):
    def __init__(self, root, image_set="val2017", image_size=320,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.root = root
        self.image_set = image_set
        self.image_size = image_size
        self.norm = transforms.Normalize(mean, std)

        ann_file = os.path.join(root, "annotations", f"instances_{image_set}.json")
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]

        img_path = os.path.join(self.root, self.image_set, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        img = np.array(image).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = self.norm(img)

        target = {
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([img_info["height"], img_info["width"]]),
        }
        return img, target


def decode_predictions(loc, scores, anchors, image_size,
                       conf_threshold=0.01, nms_threshold=0.5,
                       top_k=200, objects_per_image=200, device="cpu"):
    boxes = torch.zeros_like(loc)
    boxes[:, 0] = loc[:, 0] * anchors[:, 2] + anchors[:, 0]
    boxes[:, 1] = loc[:, 1] * anchors[:, 3] + anchors[:, 1]
    boxes[:, 2] = torch.exp(loc[:, 2].clamp(-10, 10)) * anchors[:, 2]
    boxes[:, 3] = torch.exp(loc[:, 3].clamp(-10, 10)) * anchors[:, 3]
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes[:, [0, 2]] *= image_size
    boxes[:, [1, 3]] *= image_size

    n_classes = scores.shape[-1]
    det_boxes, det_scores, det_labels = [], [], []

    for c in range(1, n_classes):
        mask = scores[:, c] > conf_threshold
        cls_scores = scores[mask, c]
        if cls_scores.size(0) == 0:
            continue
        cls_boxes = boxes[mask]
        n = min(top_k, cls_scores.size(0))
        cls_scores, idx = cls_scores.topk(n)
        cls_boxes = cls_boxes[idx]

        det_scores.append(cls_scores)
        det_boxes.append(cls_boxes)
        det_labels.append(torch.full_like(cls_scores, c, dtype=torch.long, device=device))

    if not det_scores:
        return {
            "boxes": torch.empty(0, 4, device=device),
            "scores": torch.empty(0, device=device),
            "labels": torch.empty(0, dtype=torch.long, device=device),
        }

    det_scores = torch.cat(det_scores)
    det_boxes = torch.cat(det_boxes)
    det_labels = torch.cat(det_labels)

    keep = batched_nms(det_boxes, det_scores, det_labels, nms_threshold)
    keep = keep[:objects_per_image]

    return {
        "boxes": det_boxes[keep],
        "scores": det_scores[keep],
        "labels": det_labels[keep],
    }


@torch.no_grad()
def evaluate(model, data_loader, device, args, label_to_catid, desc=""):
    model.eval()
    results = []
    total_time = 0.0
    n_images = 0

    for images, targets in data_loader:
        images = images.to(device)
        B = images.size(0)

        t0 = time.perf_counter()
        outputs = model(images)
        total_time += time.perf_counter() - t0
        n_images += B

        scores = F.softmax(outputs["scores"], dim=-1)
        boxes = outputs["boxes"]
        anchors = outputs["anchors"]

        for b in range(B):
            decoded = decode_predictions(
                boxes[b], scores[b], anchors[b],
                image_size=args.image_size,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                top_k=args.top_k,
                objects_per_image=args.objects_per_image,
                device=device,
            )
            img_id = targets[b]["image_id"].item()
            for box, score, label in zip(
                decoded["boxes"], decoded["scores"], decoded["labels"]
            ):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                cat_id = label_to_catid.get(label.item(), label.item())
                results.append({
                    "image_id": img_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4),
                    "category_id": cat_id,
                })

    avg_time = total_time / max(1, n_images) * 1000
    print(f"      {n_images} images in {total_time:.2f}s ({avg_time:.1f} ms/img)")

    if not results:
        print("      WARNING: no detections!")
        return {}

    coco = data_loader.dataset.coco
    evaluator = COCOEvaluator(coco)
    eval_out = evaluator.evaluate(results)
    metrics = eval_out["metrics"]

    error_analyzer = DetectionErrorAnalyzer(coco)
    error_metrics = error_analyzer.analyze(results)
    metrics.update(error_metrics)

    small_analyzer = SmallObjectAnalyzer(coco, args.image_size)
    small_metrics = small_analyzer.analyze(results)
    metrics.update(small_metrics)

    metrics["inference_time_ms"] = round(avg_time, 2)
    metrics["n_detections"] = len(results)

    return {"metrics": metrics, "per_category_AP": eval_out.get("per_category_AP", {})}


def print_ablation_table(all_metrics, all_names):
    std_keys = [
        ("AP", "AP"), ("AP50", "AP50"), ("AP75", "AP75"),
        ("AP_small", "AP_s"), ("AP_medium", "AP_m"), ("AP_large", "AP_l"),
        ("AR1", "AR1"), ("AR10", "AR10"), ("AR100", "AR100"),
        ("AR_small", "AR_s"), ("AR_medium", "AR_m"), ("AR_large", "AR_l"),
    ]
    extra_keys = [
        "AP_tiny", "AP_extreme_tiny", "AP75_small",
        "loc_error", "bg_confusion",
        "AR_tiny", "AR_extreme",
    ]

    n_cols = len(all_metrics) + 1
    col_widths = [10] + [18] * (n_cols - 1)

    header = f"{'Metric':<10}"
    for name in all_names:
        header += f" {name:<18}"
    header += f" {'Apple Off.':<18}"

    sep = "  " + "-" * (10 + 18 * n_cols)

    print(f"\n{'=' * 80}")
    print(f"  Ablation: Ghost Conv MobileViT-S SSD — COCO val2017")
    print(f"{'=' * 80}")
    print(header)
    print(sep)

    for key, label in std_keys:
        row = f"  {label:<10}"
        vals = []
        for m in all_metrics:
            v = m.get(key, 0.0)
            vals.append(v)
            row += f" {v:<18.4f}"
        apple_val = APPLE_OFFICIAL_METRICS.get(key, 0.0)
        row += f" {apple_val:<18.4f}"
        if len(vals) >= 2:
            delta_ghost = vals[-1] if len(vals) > 2 else vals[1]
            delta_std = vals[0]
            d = (delta_ghost if len(vals) <= 2 else vals[1]) - delta_std
            row += f"  Δ={d:+.4f}"
        print(row)

    print(sep)
    extra_row = f"  {'Params':<10}"
    for m in all_metrics:
        extra_row += f" {m.get('n_params', 0):<18,}"
    print(extra_row)

    print(f"\n  {'Extra Metrics':<12}")
    for key in extra_keys:
        found = False
        for m in all_metrics:
            if key in m:
                found = True
                break
        if found:
            print(f"    {key:<20}", end="")
            for m in all_metrics:
                v = m.get(key, 0.0)
                print(f" {v:<18.4f}", end="")
            print()

    print(f"\n{APPLE_REFERENCE_DESC}")


def save_results(all_results, all_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "experiment": "Ghost Conv MobileViT-S SSD Ablation",
        "dataset": "COCO val2017",
        "image_size": 320,
        "apple_official_reference": APPLE_OFFICIAL_METRICS,
        "apple_official_source": APPLE_REFERENCE_DESC.strip(),
        "variants": {},
    }

    for name, result in zip(all_names, all_results):
        metrics = result.get("metrics", {})
        summary["variants"][name] = {
            "metrics": {k: v for k, v in metrics.items()
                       if not isinstance(v, dict) and not isinstance(v, list)},
        }

    path = os.path.join(output_dir, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {path}")

    for name, result in zip(all_names, all_results):
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path_var = os.path.join(output_dir, f"results_{safe}.json")
        with open(path_var, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Detail: {path_var}")


def main():
    args = parse_args()

    if not _HAS_COCO:
        print("ERROR: pycocotools required. pip install pycocotools")
        sys.exit(1)
    if not args.checkpoint and not args.apple_weights and not args.checkpoint_std:
        print("ERROR: need --checkpoint, --checkpoint-std, or --apple-weights")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")

    use_ghost = args.use_ghost
    if use_ghost is None and args.no_ghost:
        use_ghost = False
    if use_ghost is None and args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "args" in state:
            use_ghost = getattr(state["args"], "use_ghost", True)
            print(f"  Inferred use_ghost={use_ghost} from checkpoint")
        else:
            use_ghost = True

    print(f"\nLoading COCO val2017 from {args.data_dir}...")
    val_dataset = COCOValDataset(
        args.data_dir, image_set="val2017", image_size=args.image_size,
    )

    if args.dry_run:
        orig_ids = val_dataset.ids
        val_dataset.ids = orig_ids[:args.dry_run_samples]
        print(f"  DRY RUN: {len(val_dataset.ids)} samples (of {len(orig_ids)})")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                              [x[1] for x in b]),
    )
    print(f"  Dataset: {len(val_dataset)} images")

    label_to_catid = build_label_to_catid(val_dataset.coco)
    print(f"  Categories: {len(label_to_catid)}")

    configs = []

    if args.checkpoint_std:
        configs.append((
            args.checkpoint_std, False, args.ghost_ratio,
            "Standard MobileViT-S"
        ))

    if args.checkpoint:
        gh = use_ghost if use_ghost is not None else True
        tag = "Ghost Conv" if gh else "Standard"
        configs.append((
            args.checkpoint, gh, args.ghost_ratio,
            f"{tag} (ratio={args.ghost_ratio})" if gh else "Standard MobileViT-S"
        ))

    if args.apple_weights:
        configs.append((
            args.apple_weights, False, 0.5,
            "Apple Weights (no Ghost)"
        ))

    all_results = []
    all_names = []

    for ckpt_path, use_gh, ratio, var_name in configs:
        print(f"\n{'─' * 60}")
        print(f"  [{var_name}]")
        model = build_model(args.n_classes, use_gh, ratio, args.image_size,
                            args.use_fpn)
        model = load_checkpoint(model, ckpt_path, use_ema=args.use_ema)
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"      Parameters: {n_params:,}")

        result = evaluate(model, val_loader, device, args, label_to_catid,
                          desc=var_name)
        if result:
            result["metrics"]["n_params"] = n_params
            all_results.append(result)
            all_names.append(var_name)

    if all_results:
        for r in all_results:
            if "metrics" in r:
                r["metrics"] = {k: v for k, v in r["metrics"].items()
                              if not isinstance(v, dict) and not isinstance(v, list)}
        print_ablation_table([r["metrics"] for r in all_results], all_names)
        save_results(all_results, all_names, args.output_dir)
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
