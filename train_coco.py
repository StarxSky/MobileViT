#!/usr/bin/env python3
"""
Train Ghost Conv MobileViT-S SSD on COCO

Optimized for P100 (12GB VRAM):
  - Ghost convolutions reduce param count by ~35%
  - Gradient checkpointing
  - Mixed precision (FP16) training
  - Batch size tuning via gradient accumulation
  - Multi-scale training (256-416)
  - EMA (Exponential Moving Average)

Usage:
  # Single GPU
  python train_coco.py --data-dir /path/to/coco --batch-size 16

  # Multi-GPU (torchrun)
  torchrun --nproc_per_node=2 train_coco.py --data-dir /path/to/coco --batch-size 16

  # Resume or load Apple weights
  python train_coco.py --resume checkpoint.pth
  python train_coco.py --load-apple-weights apple_ssd_mobilevit.pt
"""

import argparse
import math
import os
import time
import json
import random
import contextlib
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, DistributedSampler
from torchvision.ops import batched_nms

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _HAS_COCO = True
except ImportError:
    _HAS_COCO = False
    print("WARNING: pycocotools not installed. COCO evaluation will not work.")

from ghost_mobilevit_ssd.mobilevit import GhostMobileViT, MOBILENETV2_CONFIG
from ghost_mobilevit_ssd.ssd import GhostMobileViTSSD, SSDLoss
from ghost_mobilevit_ssd.utils import load_apple_pretrained_weights
from ghost_mobilevit_ssd.metrics import (
    COCOEvaluator, SmallObjectAnalyzer, DetectionErrorAnalyzer, MetricsTracker,
)
from ghost_mobilevit_ssd.visualization import (
    visualize_predictions, plot_metrics_history, plot_error_analysis,
    plot_small_object_analysis, visualize_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ghost Conv MobileViT-S SSD on COCO")
    parser.add_argument("--data-dir", type=str, default="./coco",
                        help="COCO dataset root")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=320,
                        help="Base image size for training")
    parser.add_argument("--multi-scale", action="store_true", default=True,
                        help="Use multi-scale training")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--use-ghost", action="store_true", default=True,
                        help="Use Ghost convolutions in backbone")
    parser.add_argument("--ghost-ratio", type=float, default=0.5,
                        help="Ghost ratio (0.5 = half channels via cheap ops)")
    parser.add_argument("--use-fpn", action="store_true", default=False,
                        help="Use FPN in SSD head")
    parser.add_argument("--grad-checkpoint", action="store_true", default=True,
                        help="Use gradient checkpointing")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--load-apple-weights", type=str, default=None,
                        help="Load Apple ml-cvnets pretrained weights")
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loading workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA")
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Enable DDP (auto-detected if LOCAL_RANK is set)")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ─── COCO Dataset ──────────────────────────────────────────────────────────────

class COCODetection(Dataset):
    def __init__(self, root, image_set="train2017", transform=None,
                 target_transform=None, n_classes=91):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.n_classes = n_classes

        ann_file = os.path.join(root, "annotations",
                                f"instances_{image_set}.json")
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        # Map COCO category IDs (non-contiguous, e.g. 1,2,3,...,90) to contiguous 1..N
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_label = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_path = os.path.join(self.root, self.image_set, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            x1, y1 = float(x), float(y)
            x2, y2 = x1 + float(w), y1 + float(h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label.get(ann["category_id"], 0))

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor([ann.get("area", 0) for ann in anns]),
            "iscrowd": torch.tensor(
                [ann.get("iscrowd", 0) for ann in anns], dtype=torch.long
            ),
            "orig_size": torch.tensor([img_info["height"], img_info["width"]]),
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


class SSDAugmentation:
    """SSD-style data augmentation: random crop, color jitter, expand,
    horizontal flip, resize."""

    def __init__(self, size=320, means=(0.485, 0.456, 0.406),
                 stds=(0.229, 0.224, 0.225)):
        self.size = size
        self.means = means
        self.stds = stds
        self.to_tensor_norm = transforms.Normalize(means, stds)

    def set_size(self, size):
        self.size = size

    def __call__(self, image, target):
        img = np.array(image).astype(np.float32) / 255.0
        boxes = target["boxes"].clone()
        labels = target["labels"].clone()
        H, W = img.shape[:2]

        # Random crop (SSD-style)
        if random.random() < 0.5 and boxes.size(0) > 0:
            img, boxes, labels = self._random_crop(img, boxes, labels)
            H, W = img.shape[:2]

        # Random horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            if boxes.size(0) > 0:
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]

        # Expand
        if random.random() < 0.5:
            img, boxes = self._expand(img, boxes, W, H)

        # Resize to current size
        new_H = new_W = self.size
        img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        if boxes.size(0) > 0:
            boxes[:, [0, 2]] *= new_W / W
            boxes[:, [1, 3]] *= new_H / H

        # Normalize
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = self.to_tensor_norm(img)

        # Normalize boxes to [0,1] and convert to [cx,cy,w,h]
        if boxes.size(0) > 0:
            boxes[:, [0, 2]] /= new_W
            boxes[:, [1, 3]] /= new_H
        target = self._boxes_to_cxcywh(boxes, labels)
        return img, target

    def _random_crop(self, img, boxes, labels):
        H, W = img.shape[:2]
        while True:
            min_iou = np.random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])

            for _ in range(50):
                w = random.uniform(0.3, 1.0) * W
                h = random.uniform(0.3, 1.0) * H
                x1 = random.uniform(0, max(0.0, W - w))
                y1 = random.uniform(0, max(0.0, H - h))

                crop_rect = torch.tensor([x1, y1, x1 + w, y1 + h])

                if boxes.size(0) == 0:
                    continue

                ious = torchvision.ops.box_iou(
                    boxes, crop_rect.unsqueeze(0)
                ).squeeze(1)

                if ious.min() >= min_iou:
                    img_cropped = img[int(y1):int(y1+h), int(x1):int(x1+w)]
                    boxes[:, 0] = boxes[:, 0].clamp(x1, x1 + w) - x1
                    boxes[:, 1] = boxes[:, 1].clamp(y1, y1 + h) - y1
                    boxes[:, 2] = boxes[:, 2].clamp(x1, x1 + w) - x1
                    boxes[:, 3] = boxes[:, 3].clamp(y1, y1 + h) - y1

                    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    if keep.sum() == 0:
                        continue
                    boxes = boxes[keep]
                    labels = labels[keep]
                    return img_cropped, boxes, labels
            return img, boxes, labels

    def _expand(self, img, boxes, orig_W, orig_H):
        ratio = random.uniform(1, 4)
        new_W = max(orig_W + 1, int(orig_W * ratio))
        new_H = max(orig_H + 1, int(orig_H * ratio))
        x_offset = random.randint(0, max(0, new_W - orig_W))
        y_offset = random.randint(0, max(0, new_H - orig_H))

        canvas = np.full((new_H, new_W, 3), self.means, dtype=np.float32)
        canvas[y_offset:y_offset+orig_H, x_offset:x_offset+orig_W] = img

        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset
        return canvas, boxes

    def _boxes_to_cxcywh(self, boxes, labels):
        if boxes.size(0) == 0:
            return {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.long),
            }
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        return {
            "boxes": torch.stack([cx, cy, w, h], dim=1),
            "labels": labels,
        }


# ─── EMA ───────────────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.lerp_(p, 1 - d)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


# ─── Warmup + Cosine LR ────────────────────────────────────────────────────────

class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6,
                 last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        return [self.min_lr + 0.5 * (base_lr - self.min_lr) *
                (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


# ─── COCO Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_coco(model, data_loader, epoch, output_dir, device, tracker=None,
                  vis_dir=None, error_analysis=True):
    """
    Evaluate model on COCO val set using COCOEvaluator.
    Returns a flat metrics dict plus saves analysis plots.
    """
    if not _HAS_COCO:
        print("pycocotools not available, skipping evaluation")
        return {}

    model.eval()
    results = []
    coco = data_loader.dataset.coco
    img_size = data_loader.dataset.transform.size

    for images, targets in data_loader:
        images = images.to(device)

        if isinstance(model, ModelEMA):
            preds = model.ema(images)
        else:
            preds = model(images)

        scores = F.softmax(preds["scores"], dim=-1)
        boxes = preds["boxes"]
        anchors = preds["anchors"]

        for b in range(images.size(0)):
            decoded = _decode_predictions(
                boxes[b], scores[b], anchors[b],
                conf_threshold=0.01, nms_threshold=0.5, top_k=200,
                objects_per_image=200, device=device,
                image_size=img_size,
            )
            img_id = targets[b]["image_id"].item()

            for box, score, label in zip(
                decoded["boxes"], decoded["scores"], decoded["labels"]
            ):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                results.append({
                    "image_id": img_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(score.item(), 4),
                    "category_id": label.item(),
                })

            # Save first few visualizations
            if b < 4 and vis_dir is not None and epoch % 5 == 0:
                _save_vis(images, targets, decoded, b, epoch, vis_dir, img_size)

    if not results:
        return {}

    # --- Evaluate using COCOEvaluator ---
    evaluator = COCOEvaluator(coco)
    eval_out = evaluator.evaluate(results)
    metrics = eval_out["metrics"]

    print(f"  mAP={metrics['AP']:.4f}  AP50={metrics['AP50']:.4f}  "
          f"AP75={metrics['AP75']:.4f}  AP_s={metrics['AP_small']:.4f}")

    # Extra: tiny / extreme tiny
    tiny_str = ""
    if "AP_tiny" in metrics:
        tiny_str = f"  AP_tiny={metrics['AP_tiny']:.4f}"
        if "AP_extreme_tiny" in metrics:
            tiny_str += f"  AP_xtiny={metrics['AP_extreme_tiny']:.4f}"
        print(f"  {tiny_str.strip()}")

    print(f"  AR1={metrics['AR1']:.4f}  AR10={metrics['AR10']:.4f}  "
          f"AR100={metrics['AR100']:.4f}")
    print(f"  AR_s={metrics['AR_small']:.4f}  AR_m={metrics['AR_medium']:.4f}  "
          f"AR_l={metrics['AR_large']:.4f}")

    per_cat = eval_out.get("per_category_AP", {})
    if per_cat:
        best_cats = sorted(per_cat.items(), key=lambda x: -x[1])[:5]
        worst_cats = sorted(per_cat.items(), key=lambda x: x[1])[:5]
        print(f"  Top-5 categories: {', '.join(f'{c}={v:.3f}' for c, v in best_cats)}")
        print(f"  Worst-5 categories: {', '.join(f'{c}={v:.3f}' for c, v in worst_cats)}")

    # --- Error analysis ---
    if error_analysis:
        analyzer = DetectionErrorAnalyzer(coco)
        error_metrics = analyzer.analyze(results)
        metrics.update(error_metrics)
        print(f"  Loc Error={error_metrics.get('loc_error', 0):.4f}  "
              f"BG Confusion={error_metrics.get('bg_confusion', 0):.4f}")
        if vis_dir:
            plot_error_analysis(
                error_metrics,
                os.path.join(vis_dir, f"error_analysis_epoch_{epoch}.jpg"),
                title=f"Error Analysis (Epoch {epoch})",
            )

    # --- Small object analysis ---
    small_analyzer = SmallObjectAnalyzer(coco, img_size)
    small_metrics = small_analyzer.analyze(results)
    metrics.update(small_metrics)
    dr = small_metrics.get("detection_rate", {})
    if dr:
        print(f"  DetRate: tiny={dr.get('tiny', 0)*100:.1f}%  "
              f"small={dr.get('small', 0)*100:.1f}%  "
              f"med={dr.get('medium', 0)*100:.1f}%  "
              f"large={dr.get('large', 0)*100:.1f}%")

    # --- Save metrics and plots ---
    with open(os.path.join(output_dir, f"eval_epoch_{epoch}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if tracker:
        tracker.log(epoch, "val", metrics)

    if vis_dir:
        plot_small_object_analysis(
            small_metrics,
            os.path.join(vis_dir, f"small_obj_analysis_epoch_{epoch}.jpg"),
            title=f"Small Object Analysis (Epoch {epoch})",
        )

    return metrics


def _save_vis(images, targets, decoded, batch_idx, epoch, vis_dir, img_size):
    """Save a single visualization of GT + predictions."""
    img = images[batch_idx].cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)

    gt_boxes = targets[batch_idx]["boxes"].cpu()
    gt_labels = targets[batch_idx]["labels"].cpu().tolist()
    gt_boxes_xyxy = []
    for box in gt_boxes:
        cx, cy, w, h = box.tolist()
        x1 = int((cx - w / 2) * img_size)
        y1 = int((cy - h / 2) * img_size)
        x2 = int((cx + w / 2) * img_size)
        y2 = int((cy + h / 2) * img_size)
        gt_boxes_xyxy.append([x1, y1, x2, y2])

    pred_boxes_xyxy = [b.tolist() for b in decoded["boxes"]]
    pred_labels = decoded["labels"].tolist()
    pred_scores = decoded["scores"].tolist()

    fname = os.path.join(vis_dir, f"vis_epoch{epoch}_img{batch_idx}.jpg")
    visualize_predictions(
        img, gt_boxes_xyxy, gt_labels,
        pred_boxes_xyxy, pred_labels, pred_scores,
        fname, img_size,
    )


def _decode_predictions(loc, scores, anchors, conf_threshold=0.01,
                        nms_threshold=0.5, top_k=200, objects_per_image=200,
                        device="cpu", image_size=320):
    boxes = torch.zeros_like(loc)
    boxes[:, 0] = loc[:, 0] * anchors[:, 2] + anchors[:, 0]
    boxes[:, 1] = loc[:, 1] * anchors[:, 3] + anchors[:, 1]
    boxes[:, 2] = torch.exp(loc[:, 2].clamp(-10, 10)) * anchors[:, 2]
    boxes[:, 3] = torch.exp(loc[:, 3].clamp(-10, 10)) * anchors[:, 3]
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    # Convert from normalized [0,1] to absolute pixel coords
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


# ─── Multi-Scale ───────────────────────────────────────────────────────────────

class MultiScaleBatchSampler(BatchSampler):
    """Per-epoch multi-scale: before each epoch, pick a random scale.
    All batches in that epoch share the same scale (no collate conflict)."""
    def __init__(self, sampler, batch_size, drop_last, sizes=(256, 288, 320, 352, 384)):
        super().__init__(sampler, batch_size, drop_last)
        self.sizes = sizes
        self.current_size = sizes[0]

    def set_epoch(self, epoch):
        self.current_size = self.sizes[epoch % len(self.sizes)]


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                device, epoch, args, ema=None):
    model.train()
    losses = defaultdict(float)
    n_batches = len(loader)
    t0 = time.time()

    optimizer.zero_grad()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # In DDP, skip gradient sync for all but the last grad_accum micro-batch
        is_last = (i + 1) % args.grad_accum == 0 or i == n_batches - 1
        if args.distributed and not is_last:
            cm = model.no_sync()
        else:
            cm = contextlib.nullcontext()

        with cm, autocast(enabled=args.amp):
            outputs = model(images)
            losses_dict = criterion(
                (outputs["boxes"], outputs["scores"], outputs["anchors"]),
                targets,
            )
            loss = losses_dict["total_loss"] / args.grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        for k, v in losses_dict.items():
            losses[k] += v.item()

        if i % 50 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            avg_total = losses.get('total_loss', 0) / max(1, i + 1)
            avg_loc = losses.get('loc_loss', 0) / max(1, i + 1)
            avg_conf = losses.get('conf_loss', 0) / max(1, i + 1)
            print(f"E{epoch} [{i}/{n_batches}] "
                  f"loss={avg_total:.4f} "
                  f"loc={avg_loc:.4f} "
                  f"conf={avg_conf:.4f} "
                  f"lr={lr:.2e} "
                  f"{elapsed:.1f}s/batch")

    scheduler.step()
    avg_losses = {k: v / n_batches for k, v in losses.items()}
    return avg_losses


def main():
    args = parse_args()

    # ── DDP setup ──────────────────────────────────────────────────────
    if "LOCAL_RANK" in os.environ:
        args.distributed = True
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        args.distributed = False
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed + args.local_rank)

    def print0(*msg):
        if args.local_rank == 0:
            print(*msg)

    os.makedirs(args.output_dir, exist_ok=True)

    print0(f"Using device: {device}  world_size={args.world_size}")

    # Build model
    model = GhostMobileViTSSD(
        n_classes=91 if _HAS_COCO else 80,
        ghost_ratio=args.ghost_ratio,
        use_ghost=args.use_ghost,
        image_size=(args.image_size, args.image_size),
        use_fpn=args.use_fpn,
    )

    print0(f"Model: Ghost MobileViT-S SSD")
    print0(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print0(f"  Ghost Conv: {args.use_ghost} (ratio={args.ghost_ratio})")

    model = model.to(device)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
        )
        raw_model = model.module
    else:
        raw_model = model

    # Load pretrained weights
    if args.load_apple_weights:
        print0(f"Loading Apple pretrained weights from {args.load_apple_weights}...")
        info = load_apple_pretrained_weights(
            raw_model, args.load_apple_weights,
            use_ghost=args.use_ghost,
            ghost_ratio=args.ghost_ratio,
        )
        print0(f"  Loaded: {len(info['loaded'])} keys")
        print0(f"  Skipped: {len(info['skipped'])} keys")
        print0(f"  Converted: {len(info['converted'])} keys")

    # Resume
    if args.resume:
        print0(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_map = checkpoint.get("best_map", 0.0)
    else:
        start_epoch = 0
        best_map = 0.0

    # Data
    if _HAS_COCO:
        train_transform = SSDAugmentation(size=args.image_size)
        train_dataset = COCODetection(
            args.data_dir, "train2017", transform=train_transform
        )

        class ValTransform:
            def __init__(self, size=320, means=(0.485, 0.456, 0.406),
                         stds=(0.229, 0.224, 0.225)):
                self.size = size
                self.norm = transforms.Normalize(means, stds)

            def __call__(self, image, target):
                img = np.array(image).astype(np.float32) / 255.0
                H, W = img.shape[:2]
                img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                boxes = target["boxes"].clone()
                labels = target["labels"].clone()
                if boxes.size(0) > 0:
                    boxes[:, [0, 2]] *= self.size / W
                    boxes[:, [1, 3]] *= self.size / H
                    boxes[:, [0, 2]] /= self.size
                    boxes[:, [1, 3]] /= self.size
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                target = {
                    "boxes": torch.stack([cx, cy, w, h], dim=1) if boxes.size(0) > 0 else boxes,
                    "labels": labels,
                    "image_id": target["image_id"],
                    "orig_size": target["orig_size"],
                }
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                img = self.norm(img)
                return img, target

        val_transform = ValTransform(size=args.image_size)
        val_dataset = COCODetection(
            args.data_dir, "val2017", transform=val_transform
        )
    else:
        raise ImportError("pycocotools is required for COCO training")

    # Distributed sampler
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True,
    ) if args.distributed else RandomSampler(train_dataset)

    ms_scale_sizes = (256, 288, 320, 352, 384) if args.multi_scale else (args.image_size,)
    batch_sampler = MultiScaleBatchSampler(
        train_sampler, args.batch_size, drop_last=True,
        sizes=ms_scale_sizes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            [x[1] for x in b],
        ),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            [x[1] for x in b],
        ),
        pin_memory=True,
    )

    # Optimizer & Scheduler
    params = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr * args.world_size,
                            weight_decay=args.weight_decay)
    scheduler = WarmupCosineLR(
        optimizer, args.warmup_epochs, args.epochs
    )
    scaler = GradScaler(enabled=args.amp)
    criterion = SSDLoss(n_classes=raw_model.n_classes)

    # EMA (on raw model)
    ema = ModelEMA(raw_model, decay=args.ema_decay) if args.use_ema else None
    if args.resume and "ema_state_dict" in checkpoint and ema:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    # Metrics tracker + visualization
    tracker = MetricsTracker(args.output_dir) if args.local_rank == 0 else None
    vis_dir = os.path.join(args.output_dir, "visualizations")
    if args.local_rank == 0:
        os.makedirs(vis_dir, exist_ok=True)

    # Train
    print0(f"Starting training from epoch {start_epoch} to {args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        # Set multi-scale / distributed epoch
        batch_sampler.set_epoch(epoch)
        train_transform.set_size(batch_sampler.current_size)
        print0(f"\n{'='*60}")
        print0(f"Epoch {epoch}/{args.epochs}  size={batch_sampler.current_size}")
        if args.distributed:
            train_sampler.set_epoch(epoch)

        losses = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, args, ema=ema,
        )
        print0(f"  Train: total={losses['total_loss']:.4f} "
               f"loc={losses['loc_loss']:.4f} conf={losses['conf_loss']:.4f}")

        # Eval (rank 0 only, using raw model or ema)
        if args.local_rank == 0 and ((epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1):
            eval_model = ema.ema if ema else raw_model
            metrics = evaluate_coco(
                eval_model, val_loader, epoch, args.output_dir,
                device, tracker=tracker, vis_dir=vis_dir,
            )
            if metrics:
                current_map = metrics.get("AP", 0.0)
                print0(f"  Eval: mAP={current_map:.4f} "
                       f"AP50={metrics.get('AP50', 0.0):.4f}")

                if current_map > best_map:
                    best_map = current_map
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "ema_state_dict": ema.state_dict() if ema else None,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_map": best_map,
                        "args": args,
                    }
                    path = os.path.join(args.output_dir, "model_best.pth")
                    torch.save(checkpoint, path)
                    print0(f"  New best model saved to {path}")

        # Save periodic (rank 0)
        if args.local_rank == 0 and (epoch + 1) % 20 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "ema_state_dict": ema.state_dict() if ema else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_map": best_map,
                "args": args,
            }
            path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, path)

    # Save final (rank 0)
    if args.local_rank == 0:
        checkpoint = {
            "epoch": args.epochs - 1,
            "model_state_dict": raw_model.state_dict(),
            "ema_state_dict": ema.state_dict() if ema else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_map": best_map,
            "args": args,
        }
        path = os.path.join(args.output_dir, "model_final.pth")
        torch.save(checkpoint, path)

        # Print training summary
        print0(f"\n{'='*60}")
        print0(f"Training complete!")
        print0(f"Best mAP (AP): {best_map:.4f}")
        if tracker:
            best_small = tracker.get_best("AP_small", mode="max")
            if best_small[0] is not None:
                print0(f"Best AP_small: {best_small[0]:.4f} at epoch {best_small[1]}")
            best_tiny = tracker.get_best("AP_tiny", mode="max")
            if best_tiny[0] is not None:
                print0(f"Best AP_tiny: {best_tiny[0]:.4f} at epoch {best_tiny[1]}")
            tracker.summary()
            # Plot metric history
            plot_metrics_history(
                tracker.history,
                os.path.join(args.output_dir, "metrics_history.jpg"),
                title="Ghost MobileViT-S SSD Training",
            )
        print0(f"\nDone! Best mAP: {best_map:.4f}")

    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
