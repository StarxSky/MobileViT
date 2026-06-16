# Ghost Conv MobileViT-S-SSD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A lightweight object detection model combining **Ghost Convolutions** with **MobileViT-S** backbone and **SSD** detection head. Compatible with Apple [ml-cvnets](https://github.com/apple/ml-cvnets) pretrained weights. Optimized for single-GPU (12 GB) training on COCO.

> **Ghost Conv** replaces standard pointwise convolutions with a primary + cheap depthwise split, reducing parameters by ~3–7% with minimal accuracy drop — enabling ablation studies on mobile-friendly detection.

---

## Features

- **Ghost Convolution** module (`GhostConv`) — plug-and-play replacement for pointwise convolutions
- **MobileViT-S** backbone with optional Ghost Conv in inverted residuals and MobileViT blocks
- **SSD** detection head with 6-scale predictions, 4 anchors/position, optional FPN
- **COCO** training pipeline: DDP, AMP, EMA, gradient accumulation, multi-scale (256–384)
- **Ablation inference script** — compare Ghost vs Standard vs Apple official results side-by-side
- **Apple ml-cvnets** weight compatibility — offline & runtime conversion tools
- **Comprehensive metrics**: 12 standard COCO + tiny/extreme tiny AP + error analysis + per-category AP

## Parameters

| Variant | Params | vs Original |
|---------|--------|-------------|
| Standard MobileViT-S SSD | 5.47 M | — |
| Ghost Conv (ratio=0.5) | 5.30 M | −3.1 % |
| Ghost Conv (ratio=0.25) | 5.04 M | −7.9 % |

## Requirements

```
torch>=1.12
torchvision>=0.13
opencv-python
numpy
Pillow
pycocotools
matplotlib              (optional, for metric plots)
```

## Installation

```bash
git clone https://github.com/your-username/ghost-mobilevit-ssd.git
cd ghost-mobilevit-ssd
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── ghost_mobilevit_ssd/              # Core library
│   ├── ghost_conv.py                 # GhostConv module
│   ├── mobilevit.py                  # GhostMobileViT backbone, MV2 blocks, Transformer
│   ├── ssd.py                        # GhostMobileViTSSD, SSDHead, SSDLoss, FPN
│   ├── metrics.py                    # COCOEvaluator, error analysis, small object analysis
│   ├── visualization.py              # Detection visualization, metric plots
│   └── utils.py                      # Apple weight loader, IoU utilities
├── train_coco.py                     # Training script (DDP, AMP, EMA, multi-scale)
├── inference_coco.py                 # Ablation inference & evaluation script
├── convert_weights.py                # Apple → Ghost weight converter
├── requirements.txt
└── README.md
```

---

## Usage

### Training

```bash
# Single GPU
python train_coco.py --data-dir /path/to/coco --batch-size 16

# 2 GPUs
torchrun --nproc_per_node=2 train_coco.py --data-dir /path/to/coco --batch-size 16

# Standard MobileViT-S (no Ghost Conv) — baseline for ablation
python train_coco.py --no-use-ghost

# Resume from checkpoint
python train_coco.py --resume outputs/model_best.pth
```

### Ablation Inference

The `inference_coco.py` script evaluates trained models on COCO val2017 and compares results with Apple's official ml-cvnets baseline.

```bash
# Evaluate a trained Ghost model
python inference_coco.py --checkpoint outputs/model_best.pth --data-dir ./coco

# Evaluate standard model as baseline
python inference_coco.py --checkpoint outputs/model_best.pth --data-dir ./coco --no-use-ghost

# Compare Ghost vs Standard side-by-side (ablation)
python inference_coco.py \
    --checkpoint outputs/ghost_model.pth \
    --checkpoint-std outputs/standard_model.pth \
    --data-dir ./coco

# Load Apple pretrained weights as external baseline
python inference_coco.py --apple-weights apple_ssd_mobilevit.pt --data-dir ./coco

# Quick validation on a subset
python inference_coco.py --checkpoint model.pth --data-dir ./coco --dry-run

# Use EMA weights (default: on)
python inference_coco.py --checkpoint model.pth --data-dir ./coco --use-ema
```

Output example:
```
================================================================================
  Ablation: Ghost Conv MobileViT-S SSD — COCO val2017
================================================================================
Metric       Ghost (ratio=0.5)    Standard              Apple Official         Δ
--------------------------------------------------------------------------------
AP                 22.1500             22.3000             22.3000           -0.1500
AP50               38.1000             38.2000             38.2000           -0.1000
AP75               21.8000             22.0000             22.0000           -0.2000
...
```

### Weight Conversion

```bash
# Runtime conversion (load Apple weights directly into Ghost model)
python train_coco.py --load-apple-weights apple_ssd_mobilevit.pt

# Offline conversion
python convert_weights.py apple_ssd_mobilevit.pt ghost_compatible.pth
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `./coco` | COCO dataset root |
| `--batch-size` | `16` | Batch size per GPU |
| `--lr` | `5e-4` | Initial learning rate (× world_size for DDP) |
| `--grad-accum` | `2` | Gradient accumulation steps |
| `--image-size` | `320` | Base image size |
| `--multi-scale` | `True` | Per-epoch multi-scale training (256→384) |
| `--use-ghost` | `True` | Enable Ghost convolutions |
| `--ghost-ratio` | `0.5` | Ghost ratio (primary / total channels) |
| `--amp` | `True` | Mixed precision (FP16) training |
| `--use-ema` | `True` | Exponential Moving Average |
| `--use-fpn` | `False` | Feature Pyramid Network in SSD head |

---

## Architecture

```
Input (3, 320, 320)
  └─ Stem: Conv(3→16, k=3, s=2)
       └─ layer_1: MV2 × 1 (16→32,  s=1)
            └─ layer_2: MV2 × 3 (32→64,  s=2)
                 └─ layer_3: MV2×1(s=2) + MobileViTBlock×2  → out_l3 (96)   s=8
                      └─ layer_4: MV2×1(s=2) + MobileViTBlock×4  → out_l4 (128)  s=16
                           └─ layer_5: MV2×1(s=2) + MobileViTBlock×3  → out_l5 (160)  s=32
                                └─ Extra: SeparableConv × 2             → os_64  (256)
                                     └─ Extra: SeparableConv × 1        → os_128 (256)
  └─ SSD Heads × 6 (strides 8, 16, 32, 64, 128)
       └─ 4 anchors/position → [cx,cy,w,h] offsets + 91-class scores
```

### Ghost Convolution

```
Input (C_in)
  ├── Primary:   Conv(C_in → C_out×ratio) → BN → SiLU
  └── Cheap:     DepthwiseConv(C_out×ratio → C_out×(1-ratio), k=3) → BN → SiLU
       └── Output = cat([primary, cheap], dim=1)
```

### MobileViT Block

```
Input
  ├── Local: Conv(3×3) → Conv(1×1 to transformer_dim)
  ├── Unfold → N × TransformerEncoder → Fold
  ├── Conv(1×1 back to in_ch)
  └── Fusion: Conv(3×3) on [residual, projected]
```

---

## Metrics

### Standard COCO (12)

| Metric | Description |
|--------|-------------|
| `AP` | mAP @ IoU=0.50:0.05:0.95 |
| `AP50` | AP @ IoU=0.50 |
| `AP75` | AP @ IoU=0.75 |
| `AP_small` | AP for area < 32² |
| `AP_medium` | AP for 32² ≤ area < 96² |
| `AP_large` | AP for area ≥ 96² |
| `AR1 / AR10 / AR100` | Avg recall @ 1/10/100 detections |
| `AR_small/medium/large` | Avg recall by size |

### Extra

| Metric | Description |
|--------|-------------|
| `AP_tiny` | AP for area < 16² (256 px²) |
| `AP_extreme_tiny` | AP for area < 10² (100 px²) |
| `loc_error` | Precision drop from IoU=0.25 to 0.75 |
| `bg_confusion` | Precision drop from IoU=0.25 to 0.50 |

### Training Outputs

```
outputs/
├── model_best.pth              Best checkpoint (by AP)
├── model_final.pth             Final checkpoint
├── checkpoint_epoch_N.pth      Periodic checkpoints (every 20)
├── eval_epoch_N.json           All metrics (flat JSON)
├── metrics.json                History of tracked metrics
├── metrics_history.jpg         Training curves
└── visualizations/
    ├── vis_epoch*_img*.jpg     GT vs predictions
    ├── error_analysis_*.jpg    FP type breakdown
    └── small_obj_analysis_*.jpg
```

### Ablation Outputs

```
ablation_results/
├── ablation_results.json            Unified comparison
├── results_ghost_conv.json          Ghost variant detail
├── results_standard_mobilevit-s.json  Standard variant detail
└── results_apple_weights_no_ghost.json  Apple baseline detail
```

---

## DDP (Multi-GPU)

Auto-detected via `LOCAL_RANK` (set by `torchrun`):

- `SyncBatchNorm` for consistent batch norm statistics
- `model.no_sync()` for efficient gradient accumulation
- `DistributedSampler` for non-overlapping data shards
- LR automatically scaled by `world_size`
- Only rank 0 runs evaluation and saves checkpoints

## Data Augmentation

1. **Random crop** — min IoU ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}
2. **Horizontal flip** — 50 % probability
3. **Expand** — random-color canvas (1–4× scale)
4. **Resize** — square to current epoch's size
5. **Normalize** — ImageNet mean/std → float tensor

Multi-scale: one size per epoch from {256, 288, 320, 352, 384}.

---

## Reference Results

| Metric | Apple Official | Standard | Ghost (r=0.5) | Δ |
|--------|---------------|----------|---------------|----|
| AP | 22.3 | — | — | — |
| AP50 | 38.2 | — | — | — |
| Params | — | 5.47 M | 5.30 M | −3.1 % |

*Apple official: MobileViT-S + SSDLite 320, COCO val2017 (Mehta & Rastegari, ICLR 2022)*

---

## Citation

```bibtex
@article{mehta2022mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={ICLR},
  year={2022}
}
```

## License

This project is released under the MIT License.
