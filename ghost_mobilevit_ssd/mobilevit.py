import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ghost_mobilevit_ssd.ghost_conv import GhostConv


def make_divisible(x: float, divisor: int = 8) -> int:
    return int(math.ceil(x / divisor) * divisor)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=False, use_norm=True,
                 use_act=True, act_name="swish"):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential()
        self.block.add_module("conv", nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        ))
        if use_norm:
            self.block.add_module("norm", nn.BatchNorm2d(out_channels))
        if use_act:
            act = nn.SiLU(inplace=True) if act_name == "swish" else nn.ReLU(inplace=True)
            self.block.add_module("act", act)

    def forward(self, x):
        return self.block(x)


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, use_norm=True, use_act=True):
        super().__init__()
        self.dw_conv = ConvLayer(
            in_channels, in_channels, kernel_size, stride,
            dilation=dilation, groups=in_channels,
            use_norm=True, use_act=False
        )
        self.pw_conv = ConvLayer(
            in_channels, out_channels, 1, 1,
            use_norm=use_norm, use_act=use_act
        )

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class GhostInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4,
                 dilation=1, ghost_ratio=0.5):
        super().__init__()
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        use_res_connect = stride == 1 and in_channels == out_channels
        self.use_res_connect = use_res_connect

        layers = []
        if expand_ratio != 1:
            layers.append(("exp_1x1", ConvLayer(
                in_channels, hidden_dim, 1, 1,
                use_norm=True, use_act=True
            )))

        layers.append(("conv_3x3", ConvLayer(
            hidden_dim, hidden_dim, 3, stride,
            dilation=dilation, groups=hidden_dim,
            use_norm=True, use_act=True
        )))

        layers.append(("red_1x1", ConvLayer(
            hidden_dim, out_channels, 1, 1,
            use_norm=True, use_act=False
        )))

        self.block = nn.Sequential()
        for name, layer in layers:
            self.block.add_module(name, layer)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class GhostInvertedResidualV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4,
                 dilation=1, ghost_ratio=0.5):
        super().__init__()
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        use_res_connect = stride == 1 and in_channels == out_channels
        self.use_res_connect = use_res_connect

        layers = []
        if expand_ratio != 1:
            layers.append(("exp_1x1", GhostConv(
                in_channels, hidden_dim, 1, 1, ratio=ghost_ratio
            )))

        layers.append(("conv_3x3", ConvLayer(
            hidden_dim, hidden_dim, 3, stride,
            dilation=dilation, groups=hidden_dim,
            use_norm=True, use_act=True
        )))

        layers.append(("red_1x1", GhostConv(
            hidden_dim, out_channels, 1, 1, ratio=ghost_ratio
        )))

        self.block = nn.Sequential()
        for name, layer in layers:
            self.block.add_module(name, layer)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        y = self.norm(x)
        y = self.attn(y, y, y)[0]
        y = self.drop(y)
        return x + y


class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_latent_dim, embed_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, num_heads=8,
                 attn_dropout=0.0, dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.ffn = TransformerFFN(embed_dim, ffn_latent_dim, dropout, ffn_dropout)

    def forward(self, x):
        x = self.mha(x)
        x = self.ffn(x)
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, ffn_dim,
                 n_transformer_blocks=2, num_heads=4, attn_dropout=0.0,
                 dropout=0.0, ffn_dropout=0.0, patch_h=2, patch_w=2,
                 conv_ksize=3, no_fusion=False, ghost_ratio=0.5,
                 use_ghost=False):
        super().__init__()

        self.local_rep = nn.Sequential()
        self.local_rep.add_module("conv_3x3", ConvLayer(
            in_channels, in_channels, conv_ksize, 1,
            groups=1, use_norm=True, use_act=True
        ))
        if use_ghost:
            self.local_rep.add_module("conv_1x1", GhostConv(
                in_channels, transformer_dim, 1, 1,
                ratio=ghost_ratio, use_norm=False, use_act=False,
            ))
        else:
            self.local_rep.add_module("conv_1x1", ConvLayer(
                in_channels, transformer_dim, 1, 1,
                use_norm=False, use_act=False,
            ))

        self.global_rep = nn.Sequential(*[
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(n_transformer_blocks)
        ])
        self.global_rep.add_module("final_norm", nn.LayerNorm(transformer_dim))

        if use_ghost:
            self.conv_proj = GhostConv(
                transformer_dim, in_channels, 1, 1,
                ratio=ghost_ratio, use_norm=True, use_act=True,
            )
        else:
            self.conv_proj = ConvLayer(
                transformer_dim, in_channels, 1, 1,
                use_norm=True, use_act=True,
            )

        self.fusion = None
        if not no_fusion:
            self.fusion = ConvLayer(
                2 * in_channels, in_channels, conv_ksize, 1,
                use_norm=True, use_act=True
            )

        self.num_heads = num_heads
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = patch_w * patch_h

    def unfolding(self, x):
        patch_w, patch_h = self.patch_w, self.patch_h
        B, C, H, W = x.shape

        new_h = int(math.ceil(H / patch_h) * patch_h)
        new_w = int(math.ceil(W / patch_w) * patch_w)

        interpolate = False
        if new_w != W or new_h != H:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                              align_corners=False)
            interpolate = True

        n_w, n_h = new_w // patch_w, new_h // patch_h
        n_patches = n_h * n_w

        reshaped = x.reshape(B * C * n_h, patch_h, n_w, patch_w)
        transposed = reshaped.transpose(1, 2)
        reshaped = transposed.reshape(B, C, n_patches, self.patch_area)
        transposed = reshaped.transpose(1, 3)
        patches = transposed.reshape(B * self.patch_area, n_patches, -1)

        info = {
            "orig_size": (H, W),
            "batch_size": B,
            "interpolate": interpolate,
            "total_patches": n_patches,
            "n_patches_w": n_w,
            "n_patches_h": n_h,
        }
        return patches, info

    def folding(self, patches, info):
        B = info["batch_size"]
        n_patches = info["total_patches"]
        n_w, n_h = info["n_patches_w"], info["n_patches_h"]

        patches = patches.contiguous().view(B, self.patch_area, n_patches, -1)
        C = patches.shape[-1]
        patches = patches.transpose(1, 3)

        fm = patches.reshape(B * C * n_h, n_w, self.patch_h, self.patch_w)
        fm = fm.transpose(1, 2)
        fm = fm.reshape(B, C, n_h * self.patch_h, n_w * self.patch_w)

        if info["interpolate"]:
            fm = F.interpolate(fm, size=info["orig_size"], mode="bilinear",
                               align_corners=False)
        return fm

    def forward(self, x):
        res = x
        fm = self.local_rep(x)
        patches, info = self.unfolding(fm)
        patches = self.global_rep(patches)
        fm = self.folding(patches, info)
        fm = self.conv_proj(fm)
        if self.fusion is not None:
            fm = self.fusion(torch.cat([res, fm], dim=1))
        return fm


def _compute_head_dim(transformer_channels, num_heads=4):
    head_dim = transformer_channels // num_heads
    remainder = transformer_channels % num_heads
    if remainder != 0:
        import math
        for hd in range(head_dim, head_dim + 16):
            if transformer_channels % hd == 0:
                return hd
    return head_dim


MOBILENETV2_CONFIG = {
    "layer1": {"out_channels": 32, "expand_ratio": 4, "num_blocks": 1, "stride": 1},
    "layer2": {"out_channels": 64, "expand_ratio": 4, "num_blocks": 3, "stride": 2},
    "layer3": {
        "out_channels": 96, "expand_ratio": 4, "num_blocks": 1, "stride": 2,
        "block_type": "mobilevit",
        "transformer_channels": 144, "ffn_dim": 288, "transformer_blocks": 2,
        "patch_h": 2, "patch_w": 2, "num_heads": 4,
    },
    "layer4": {
        "out_channels": 128, "expand_ratio": 4, "num_blocks": 1, "stride": 2,
        "block_type": "mobilevit",
        "transformer_channels": 192, "ffn_dim": 384, "transformer_blocks": 4,
        "patch_h": 2, "patch_w": 2, "num_heads": 4,
    },
    "layer5": {
        "out_channels": 160, "expand_ratio": 4, "num_blocks": 1, "stride": 2,
        "block_type": "mobilevit",
        "transformer_channels": 240, "ffn_dim": 480, "transformer_blocks": 3,
        "patch_h": 2, "patch_w": 2, "num_heads": 4,
    },
    "last_layer_exp_factor": 4,
}


def _make_mobilenet_layer(cfg, in_channels, ghost_ratio=1.0, use_ghost=False):
    out_channels = cfg["out_channels"]
    num_blocks = cfg["num_blocks"]
    expand_ratio = cfg["expand_ratio"]
    stride = cfg.get("stride", 1)

    block_cls = GhostInvertedResidualV2 if use_ghost else GhostInvertedResidual
    layers = []
    for i in range(num_blocks):
        s = stride if i == 0 else 1
        layers.append(block_cls(
            in_channels, out_channels, stride=s,
            expand_ratio=expand_ratio, ghost_ratio=ghost_ratio
        ))
        in_channels = out_channels
    return nn.Sequential(*layers), in_channels


def _make_mit_layer(cfg, in_channels, ghost_ratio=0.5, use_ghost=False):
    stride = cfg.get("stride", 1)
    out_channels = cfg["out_channels"]

    block_cls = GhostInvertedResidualV2 if use_ghost else GhostInvertedResidual
    layers = []

    if stride == 2:
        layers.append(block_cls(
            in_channels, out_channels, stride=stride,
            expand_ratio=cfg.get("expand_ratio", 4), ghost_ratio=ghost_ratio
        ))
        in_channels = out_channels

    head_dim = _compute_head_dim(cfg["transformer_channels"], cfg.get("num_heads", 4))
    layers.append(MobileViTBlock(
        in_channels=in_channels,
        transformer_dim=cfg["transformer_channels"],
        ffn_dim=cfg["ffn_dim"],
        n_transformer_blocks=cfg["transformer_blocks"],
        patch_h=cfg["patch_h"],
        patch_w=cfg["patch_w"],
        num_heads=cfg.get("num_heads", 4),
    ))

    return nn.Sequential(*layers), in_channels


class GhostMobileViT(nn.Module):
    def __init__(self, ghost_ratio=0.5, use_ghost=False, num_classes=1000):
        super().__init__()
        self.use_ghost = use_ghost
        self.ghost_ratio = ghost_ratio

        image_channels = 3
        out_channels = 16
        cfg = MOBILENETV2_CONFIG

        self.model_conf_dict = {}

        self.conv_1 = ConvLayer(image_channels, out_channels, 3, stride=2)
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}
        in_channels = out_channels

        self.layer_1, out_channels = _make_mobilenet_layer(
            cfg["layer1"], in_channels, ghost_ratio, use_ghost
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels

        self.layer_2, out_channels = _make_mobilenet_layer(
            cfg["layer2"], in_channels, ghost_ratio, use_ghost
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels

        self.layer_3, out_channels = _make_mit_layer(
            cfg["layer3"], in_channels, ghost_ratio, use_ghost
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels

        self.layer_4, out_channels = _make_mit_layer(
            cfg["layer4"], in_channels, ghost_ratio, use_ghost
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels

        self.layer_5, out_channels = _make_mit_layer(
            cfg["layer5"], in_channels, ghost_ratio, use_ghost
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels

        self.conv_1x1_exp = None
        if use_ghost:
            exp_channels = min(cfg["last_layer_exp_factor"] * in_channels, 960)
            self.conv_1x1_exp = GhostConv(
                in_channels, exp_channels, 1, 1, ratio=ghost_ratio
            )
        else:
            exp_channels = min(cfg["last_layer_exp_factor"] * in_channels, 960)
            self.conv_1x1_exp = ConvLayer(
                in_channels, exp_channels, 1, 1,
                use_norm=True, use_act=True
            )
        self.model_conf_dict["exp_before_cls"] = {"in": in_channels, "out": exp_channels}

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(exp_channels, num_classes, bias=True),
        )

    def extract_end_points_all(self, x, use_l5=True, use_l5_exp=False):
        out = {}
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        out["out_l2"] = x
        x = self.layer_3(x)
        out["out_l3"] = x
        x = self.layer_4(x)
        out["out_l4"] = x
        if use_l5:
            x = self.layer_5(x)
            out["out_l5"] = x
            if use_l5_exp and self.conv_1x1_exp is not None:
                out["out_l5_exp"] = self.conv_1x1_exp(x)
        return out

    def extract_features(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        if self.conv_1x1_exp is not None:
            x = self.conv_1x1_exp(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)
        return x
