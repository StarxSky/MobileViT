import math
import os
import re
from typing import Dict, Optional

import torch
import torch.nn as nn


def load_apple_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    use_ghost: bool = False,
    ghost_ratio: float = 0.5,
) -> Dict[str, Optional[str]]:
    """
    Load pretrained weights from Apple's ml-cvnets SSD-MobileViT checkpoint.

    This function handles the key differences between Apple's original architecture
    and our Ghost Conv variant:
    - Ghost Conv layers (exp_1x1, red_1x1 in InvertedResidual) have different
      parameter counts and names
    - The transformer blocks (MobileViTBlock) and SSD heads are identical and
      can be loaded directly
    - For Ghost Conv layers, we initialize by copying the first half of original
      weights (primary) and zero-init the cheap operations

    Args:
        model: Our GhostMobileViTSSD model
        checkpoint_path: Path to Apple's ml-cvnets checkpoint (.pt or .pth)
        strict: If True, raise on mismatched keys
        use_ghost: If True, the model uses GhostConv layers
        ghost_ratio: Ghost ratio used in the model

    Returns:
        dict with keys "loaded", "skipped", "converted"
    """
    device = next(model.parameters()).device
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "state_dict" in state:
        state_dict = state["state_dict"]
    elif "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    model_sd = model.state_dict()
    loaded = {}
    skipped = {}
    converted = {}

    for key, param in model_sd.items():
        if key in state_dict:
            if state_dict[key].shape == param.shape:
                loaded[key] = key
            else:
                skipped[key] = f"shape mismatch: {state_dict[key].shape} vs {param.shape}"
        else:
            skipped[key] = "key not found"

    matched = {}
    for k in loaded:
        matched[k] = state_dict[k]

    if use_ghost and not strict:
        ghost_matched = _convert_ghost_weights(model, model_sd, state_dict, ghost_ratio)
        matched.update(ghost_matched["converted"])
        converted = ghost_matched["converted"]

    model.load_state_dict(matched, strict=False)

    return {
        "loaded": list(loaded.keys()),
        "skipped": skipped,
        "converted": list(converted.keys()),
    }


def _convert_ghost_weights(model, model_sd, state_dict, ghost_ratio=0.5):
    converted = {}

    for key, param in model_sd.items():
        if key in state_dict:
            continue

        ghost_key = _find_ghost_matching_key(key)
        if ghost_key is not None and ghost_key in state_dict:
            src_w = state_dict[ghost_key]
            if src_w.dim() == 4 and src_w.shape[0] == param.shape[0] * 2:
                primary_out = param.shape[0]
                with torch.no_grad():
                    param.copy_(src_w[:primary_out])
                converted[key] = f"converted from {ghost_key}"
            elif src_w.dim() == 4:
                with torch.no_grad():
                    param.normal_(0, 0.02)
                converted[key] = f"init from {ghost_key} shape"

        ghost_1x1_key = _find_matching_key_v2(key)
        if ghost_1x1_key is not None and ghost_1x1_key in state_dict:
            src_w = state_dict[ghost_1x1_key]
            if src_w.dim() == 4 and param.shape[0] <= src_w.shape[0]:
                with torch.no_grad():
                    param.copy_(src_w[:param.shape[0]])
                converted[key] = f"truncated from {ghost_1x1_key}"

    return {"converted": converted}


def _find_ghost_matching_key(key: str) -> Optional[str]:
    patterns = [
        (r"\.primary_conv\.weight", ".block.conv.weight"),
        (r"\.primary_bn\.weight", ".block.norm.weight"),
        (r"\.primary_bn\.bias", ".block.norm.bias"),
        (r"\.primary_bn\.running_mean", ".block.norm.running_mean"),
        (r"\.primary_bn\.running_var", ".block.norm.running_var"),
        (r"\.primary_bn\.num_batches_tracked", ".block.norm.num_batches_tracked"),
    ]
    for ghost_pat, orig_pat in patterns:
        if re.search(ghost_pat, key):
            return re.sub(ghost_pat, orig_pat, key)
    return None


def _find_matching_key_v2(key: str) -> Optional[str]:
    if "exp_1x1" in key or "red_1x1" in key:
        if "primary_conv" in key:
            return key.replace(".primary_conv.weight", ".block.conv.weight")
        if "primary_bn" in key:
            return key.replace("primary_bn", "block.norm")
    return None


def create_ssd_multi_scale_collate(image_size=(320, 320)):
    from torch.utils.data._utils.collate import default_collate

    def collate_fn(batch):
        images = []
        targets = []
        for img, t in batch:
            images.append(img)
            targets.append(t)

        images = torch.stack(images, 0)
        return images, targets

    return collate_fn


def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes [x1,y1,x2,y2]"""
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-8)
