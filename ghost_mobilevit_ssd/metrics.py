import json
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch


class COCOEvaluator:
    """Enhanced COCO evaluator returning all 12 standard metrics plus
    per-category AP and extra small-object ranges."""

    def __init__(self, coco_gt, iou_type="bbox"):
        self.coco_gt = coco_gt
        self.iou_type = iou_type

    def evaluate(self, results):
        if not results:
            return _empty_metrics()

        from pycocotools.cocoeval import COCOeval

        coco_dt = self.coco_gt.loadRes(results)

        # --- Standard evaluation (all 12 stats) ---
        stats12, per_cat, stats_extra = _run_coco_eval(self.coco_gt, coco_dt, iou_type=self.iou_type)

        metrics = {
            "AP": stats12[0],
            "AP50": stats12[1],
            "AP75": stats12[2],
            "AP_small": stats12[3],
            "AP_medium": stats12[4],
            "AP_large": stats12[5],
            "AR1": stats12[6],
            "AR10": stats12[7],
            "AR100": stats12[8],
            "AR_small": stats12[9],
            "AR_medium": stats12[10],
            "AR_large": stats12[11],
        }

        if stats_extra:
            metrics.update(stats_extra)

        per_cat_metrics = {}
        for cat_id, ap in per_cat.items():
            cat_name = self.coco_gt.loadCats([cat_id])[0]["name"]
            per_cat_metrics[cat_name] = ap

        return {
            "metrics": metrics,
            "per_category_AP": per_cat_metrics,
        }


def _run_coco_eval(coco_gt, coco_dt, iou_type="bbox"):
    """Run COCOeval and return (stats12, per_category_ap, extra_small_stats)."""
    from pycocotools.cocoeval import COCOeval

    # Standard eval
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    stats12 = coco_eval.stats.tolist()

    # Per-category AP (AP@IoU=0.50:0.95, area=all, maxDets=100)
    per_cat = {}
    cat_ids = coco_gt.getCatIds()
    for i, cat_id in enumerate(cat_ids):
        precision = coco_eval.eval["precision"]
        # precision[T, R, K, A, M]  — T=10 IoU, R=101 recall, K=cat, A=area, M=maxDets
        if precision.shape[2] > i:
            ap = precision[:, :, i, 0, -1]  # all area, maxDets=100
            ap = ap[ap > -1]
            per_cat[cat_id] = float(ap.mean()) if ap.size > 0 else 0.0
        else:
            per_cat[cat_id] = 0.0

    # Extra small-object ranges (tiny < 16², extreme < 10²)
    extra = {}
    for name, area_rng in [("AP_tiny", [0, 16 ** 2]),
                           ("AP_extreme_tiny", [0, 10 ** 2])]:
        if area_rng[1] <= area_rng[0]:
            continue
        e = COCOeval(coco_gt, coco_dt, iou_type)
        e.params.areaRng = [area_rng, area_rng]
        e.params.maxDets = [100]
        e.evaluate()
        e.accumulate()
        ap = e.stats[0]
        extra[name] = float(ap) if not np.isnan(ap) else 0.0

    # Also compute AP for small objects at higher IoU (0.75) for diagnosis
    e75 = COCOeval(coco_gt, coco_dt, iou_type)
    e75.params.iouThrs = np.array([0.75])
    e75.params.areaRng = [[0, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    e75.evaluate()
    e75.accumulate()
    extra["AP75_small"] = float(e75.stats[0]) if not np.isnan(e75.stats[0]) else 0.0
    extra["AP75_medium"] = float(e75.stats[1]) if not np.isnan(e75.stats[1]) else 0.0
    extra["AP75_large"] = float(e75.stats[2]) if not np.isnan(e75.stats[2]) else 0.0

    return stats12, per_cat, extra


class DetectionErrorAnalyzer:
    """Analyze false positive types and localization errors.
    Mirrors the COCO detection error analysis (C75, C50, Loc, Sim, Oth, BG)."""

    def __init__(self, coco_gt):
        self.coco_gt = coco_gt

    def analyze(self, results):
        if not results:
            return {}

        from pycocotools.cocoeval import COCOeval

        coco_dt = self.coco_gt.loadRes(results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        precision = coco_eval.eval["precision"]

        # Mean precision at different IoU thresholds (all categories, all areas)
        ap_C75 = _mean_precision_at_iou(precision, iou_idx=8)   # IoU=0.75
        ap_C50 = _mean_precision_at_iou(precision, iou_idx=5)   # IoU=0.50
        ap_C25 = _mean_precision_at_iou(precision, iou_idx=2)   # IoU=0.25

        # Localization error: drop in precision from loose to strict IoU
        loc_error = max(0.0, ap_C25 - ap_C75) if ap_C25 > 0 else 0.0

        # Background confusion: precision at very low IoU minus precision at 0.5
        bg_confusion = max(0.0, ap_C25 - ap_C50) if ap_C25 > 0 else 0.0

        return {
            "AP_C75": float(ap_C75),
            "AP_C50": float(ap_C50),
            "AP_C25": float(ap_C25),
            "loc_error": float(loc_error),
            "bg_confusion": float(bg_confusion),
        }


def _mean_precision_at_iou(precision, iou_idx):
    """Mean precision across all categories/areas at given IoU threshold index."""
    if iou_idx >= precision.shape[0]:
        return 0.0
    p = precision[iou_idx, :, :, 0, -1]  # IoU=T, recall=R, cat=K, area=all, max=100
    p = p[p > -1]
    return float(p.mean()) if p.size > 0 else 0.0


class SmallObjectAnalyzer:
    """Detailed analysis of small object detection performance.
    Computes per-size detection rates, recall breakdown, and
    detection difficulty analysis."""

    def __init__(self, coco_gt, img_size=320):
        self.coco_gt = coco_gt
        self.img_size = img_size
        # Pre-compute ground-truth area distribution for the dataset
        self._gt_area_stats = None

    def compute_gt_stats(self):
        """Compute ground truth area distribution in absolute pixels."""
        ann_ids = self.coco_gt.getAnnIds()
        anns = self.coco_gt.loadAnns(ann_ids)
        sizes = {"tiny": 0, "small": 0, "medium": 0, "large": 0, "total": len(anns)}
        for ann in anns:
            bbox = ann["bbox"]
            area = bbox[2] * bbox[3]
            if area < 16 * 16:
                sizes["tiny"] += 1
            elif area < 32 * 32:
                sizes["small"] += 1
            elif area < 96 * 96:
                sizes["medium"] += 1
            else:
                sizes["large"] += 1
        self._gt_area_stats = sizes
        return sizes

    def analyze(self, results):
        """Analyze detection results by object size."""
        if not results:
            return {}

        from pycocotools.cocoeval import COCOeval

        coco_dt = self.coco_gt.loadRes(results)

        # Compute detection rate per size class
        sizes_gt = self._gt_area_stats or self.compute_gt_stats()

        dets_by_size = {"tiny": 0, "small": 0, "medium": 0, "large": 0, "total": 0}
        for r in results:
            w, h = r["bbox"][2], r["bbox"][3]
            area = w * h
            dets_by_size["total"] += 1
            if area < 16 * 16:
                dets_by_size["tiny"] += 1
            elif area < 32 * 32:
                dets_by_size["small"] += 1
            elif area < 96 * 96:
                dets_by_size["medium"] += 1
            else:
                dets_by_size["large"] += 1

        detection_rate = {}
        for key in ["tiny", "small", "medium", "large"]:
            gt = sizes_gt.get(key, 0)
            det = dets_by_size.get(key, 0)
            detection_rate[key] = det / max(1, gt)

        # Recall at different area ranges using COCOeval
        extra_recall = {}
        for name, area_rng in [("AR_tiny", [0, 16 ** 2]),
                               ("AR_extreme", [0, 10 ** 2])]:
            e = COCOeval(self.coco_gt, coco_dt, "bbox")
            e.params.areaRng = [area_rng, area_rng]
            e.params.maxDets = [100]
            e.evaluate()
            e.accumulate()
            ar = e.stats[8]  # AR@100
            extra_recall[name] = float(ar) if not np.isnan(ar) else 0.0

        return {
            "gt_distribution": sizes_gt,
            "det_distribution": dets_by_size,
            "detection_rate": detection_rate,
            **extra_recall,
        }


class MetricsTracker:
    """Track metrics across epochs, persist to JSON and CSV."""

    def __init__(self, save_dir, filename="metrics.json"):
        self.save_dir = save_dir
        self.filename = filename
        self.path = os.path.join(save_dir, filename)
        self.history = []
        os.makedirs(save_dir, exist_ok=True)

    def log(self, epoch, split, metrics_dict):
        entry = {"epoch": epoch, "split": split}
        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    entry[f"{k}_{sub_k}"] = float(sub_v) if not isinstance(sub_v, (int, float)) else sub_v
            else:
                entry[k] = float(v) if not isinstance(v, (int, float)) else v
        self.history.append(entry)
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_best(self, metric="AP", mode="max"):
        best = None
        best_epoch = None
        for entry in self.history:
            if metric in entry:
                val = entry[metric]
                if best is None or (mode == "max" and val > best) or (mode == "min" and val < best):
                    best = val
                    best_epoch = entry["epoch"]
        return best, best_epoch

    def summary(self):
        """Print a summary of all tracked metrics."""
        if not self.history:
            print("No metrics tracked yet.")
            return
        print(f"{'Metric':<25} {'Best':<10} {'Latest':<10} {'At epoch':<10}")
        print("-" * 60)
        keys = list(self.history[0].keys())
        keys = [k for k in keys if k not in ("epoch", "split")]
        seen = set()
        for k in keys:
            base = k.split("_per_category")[0].split("_gt_distribution")[0].split("_det_distribution")[0].split("_detection_rate")[0]
            if base in seen:
                continue
            seen.add(base)
            vals = [e.get(k, None) for e in self.history if e.get(k) is not None]
            if not vals:
                continue
            # Find best
            best_val = max(vals)
            best_epoch = next((e["epoch"] for e in self.history if e.get(k) == best_val), "?")
            latest = vals[-1]
            print(f"{k:<25} {best_val:<10.4f} {latest:<10.4f} {best_epoch:<10}")

    def plot(self, metrics=None, save_path=None):
        """Plot metric curves over epochs using matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ["AP", "AP50", "AP_small", "AP_medium", "AP_large"]
        keys = [m for m in metrics if any(m in e for e in self.history)]
        if not keys:
            return

        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, key in zip(axes, keys):
            epochs = []
            vals = []
            for entry in self.history:
                if key in entry:
                    epochs.append(entry["epoch"])
                    vals.append(entry[key])
            if not epochs:
                continue
            ax.plot(epochs, vals, "-o", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.close(fig)


def _empty_metrics():
    return {
        "metrics": {k: 0.0 for k in [
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
            "AP_tiny", "AP_extreme_tiny", "AP75_small", "AP75_medium", "AP75_large",
        ]},
        "per_category_AP": {},
    }
