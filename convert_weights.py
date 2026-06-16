#!/usr/bin/env python3
"""
Convert Apple ml-cvnets SSD-MobileViT pretrained weights for use with
Ghost Conv MobileViT-S SSD.

Usage:
  # Load Apple weights into Ghost model (runtime conversion)
  python train_coco.py --load-apple-weights /path/to/apple_ssd_mobilevit.pt

  # Convert weights file to Ghost-compatible format
  python convert_weights.py apple_ssd_mobilevit.pt ghost_compatible.pth

The conversion handles:
  - Standard Conv → GhostConv weight splitting:
    primary_conv.weight = original.weight[:out//2]
    (cheap ops are initialized to near-zero)
  - Direct mapping for identical layers (Transformer, SSD heads)
  - BatchNorm statistics mapping
"""

import argparse
import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Apple ml-cvnets SSD-MobileViT weights to Ghost Conv format"
    )
    parser.add_argument("input", type=str, help="Input Apple checkpoint (.pt/.pth)")
    parser.add_argument("output", type=str, nargs="?", default=None,
                        help="Output path for converted weights")
    parser.add_argument("--ghost-ratio", type=float, default=0.5,
                        help="Ghost ratio used in the target model")
    parser.add_argument("--show-mapping", action="store_true",
                        help="Show detailed key mapping")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on unmapped keys")
    return parser.parse_args()


# MobileViTBlock - same state dict keys for identical layers
# SSD heads - same state dict keys
# Extra layers - same state dict keys


def is_ghost_layer(key: str) -> bool:
    """Check if a key corresponds to a GhostConv layer.

    GhostConv layers appear in:
    - InvertedResidual: exp_1x1, red_1x1
    - MobileViTBlock: conv_1x1, conv_proj
    They have a 'block' submodule with 'conv'/'norm'/'act' (same as ConvLayer)
    AND a 'cheap' submodule with 'conv'/'norm'/'act' (the extra cheap ops).
    """
    return ".cheap." in key


def find_original_key_for_ghost(key: str, ghost_ratio: float = 0.5) -> Optional[str]:
    """Given a GhostConv key, find the corresponding original key.

    GhostConv stores primary branch at block.conv / block.norm (same as ConvLayer),
    so the original ConvLayer key is identical. We just need to verify the shapes
    are compatible.
    """
    if is_ghost_layer(key):
        return None  # cheap layers have no original counterpart
    return key


def convert_checkpoint(input_path: str, output_path: str = None,
                       ghost_ratio: float = 0.5, show_mapping: bool = False,
                       strict: bool = False) -> OrderedDict:
    """
    Convert Apple ml-cvnets checkpoint to Ghost Conv compatible format.

    Strategy:
    - For identical layers (Transformer, SSD heads, extra_layers, conv_3x3):
      copy weights directly (key names already match)
    - For Ghost Conv layers (exp_1x1, red_1x1, conv_1x1, conv_proj):
      - block.conv.weight ← original conv.weight[:primary_out] (truncated)
      - block.norm.* ← original norm.* (truncated if needed)
      - cheap.* layers are initialized to near-zero (represent identity)
    - Apple layers with no ghost counterpart: dropped
    - Ghost layers with no Apple counterpart: initialized fresh
    """
    print(f"Loading Apple checkpoint: {input_path}")
    state = torch.load(input_path, map_location="cpu", weights_only=True)

    if "state_dict" in state:
        apple_sd = state["state_dict"]
    elif "model_state_dict" in state:
        apple_sd = state["model_state_dict"]
    else:
        apple_sd = state

    # Build ghost model state dict
    from ghost_mobilevit_ssd.ssd import GhostMobileViTSSD

    ghost_model = GhostMobileViTSSD(
        n_classes=91, ghost_ratio=ghost_ratio, use_ghost=True
    )
    ghost_sd = ghost_model.state_dict()

    converted = OrderedDict()
    stats = {"direct": 0, "ghost": 0, "init_ghost": 0, "shape_mismatch": 0}

    for key in ghost_sd:
        dst = ghost_sd[key]

        if key in apple_sd:
            src = apple_sd[key]
            if src.shape == dst.shape:
                converted[key] = src
                stats["direct"] += 1
                if show_mapping:
                    print(f"  DIRECT: {key}")
            elif src.dim() == dst.dim() and src.dim() >= 1:
                # Shape mismatch — likely GhostConv primary branch
                if src.shape[0] >= dst.shape[0]:
                    # Truncate first N output channels
                    slc = [slice(None)] * src.dim()
                    slc[0] = slice(0, dst.shape[0])
                    converted[key] = src[tuple(slc)].contiguous()
                    stats["ghost"] += 1
                    if show_mapping:
                        print(f"  TRUNC:  {key} {tuple(src.shape)}→{tuple(dst.shape)}")
                elif dst.shape[0] >= src.shape[0]:
                    # Pad with zeros
                    pad = [0] * (src.dim() * 2)
                    pad[-1] = dst.shape[0] - src.shape[0]
                    converted[key] = torch.nn.functional.pad(src, pad)
                    stats["ghost"] += 1
                else:
                    converted[key] = dst
                    stats["shape_mismatch"] += 1
            else:
                converted[key] = dst
                stats["shape_mismatch"] += 1
        elif is_ghost_layer(key):
            # cheap branch — initialize to identity near-zero
            if "weight" in key:
                converted[key] = torch.zeros_like(dst)
                if ".0." in key:
                    converted[key].normal_(0, 0.001)
                elif ".1." in key:
                    converted[key].fill_(1.0)
            elif "bias" in key:
                converted[key] = torch.zeros_like(dst)
            elif "running_mean" in key:
                converted[key] = torch.zeros_like(dst)
            elif "running_var" in key:
                converted[key] = torch.ones_like(dst)
            elif "num_batches_tracked" in key:
                converted[key] = dst
            else:
                converted[key] = dst
            stats["ghost"] += 1
            if show_mapping:
                print(f"  CHEAP:  {key}")
        else:
            converted[key] = dst
            stats["init_ghost"] += 1
            if show_mapping:
                print(f"  INIT:   {key} (not in Apple checkpoint)")

    # Report
    print(f"\nConversion complete:")
    print(f"  Direct matches:    {stats['direct']}")
    print(f"  Ghost converted:   {stats['ghost']}")
    print(f"  Ghost initialized: {stats['init_ghost']}")
    print(f"  Total:             {len(converted)}")

    # Verify
    missing, unexpected = ghost_model.load_state_dict(converted, strict=strict)

    if missing:
        print(f"\nMissing keys: {len(missing)}")
        for k in missing:
            print(f"  - {k}")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    if output_path:
        print(f"\nSaving converted checkpoint to: {output_path}")
        torch.save({"state_dict": converted}, output_path)

    return converted


def main():
    args = parse_args()
    output_path = args.output or args.input.replace(".pt", "_ghost.pt") \
                                            .replace(".pth", "_ghost.pth")
    convert_checkpoint(
        args.input, output_path,
        ghost_ratio=args.ghost_ratio,
        show_mapping=args.show_mapping,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
