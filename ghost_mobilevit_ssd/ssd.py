import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ghost_mobilevit_ssd.mobilevit import (
    GhostMobileViT, ConvLayer, SeparableConv, GhostConv
)


class SSDHead(nn.Module):
    def __init__(self, in_channels, n_classes, n_anchors, n_coordinates=4,
                 proj_channels=-1, kernel_size=3, stride=1):
        super().__init__()
        self.proj_layer = None
        self.proj_channels = None

        if proj_channels != -1 and proj_channels != in_channels and kernel_size > 1:
            self.proj_layer = ConvLayer(
                in_channels, proj_channels, 1, 1,
                use_norm=True, use_act=True
            )
            in_channels = proj_channels
            self.proj_channels = proj_channels

        conv_fn = ConvLayer if kernel_size == 1 else SeparableConv
        if kernel_size > 1 and stride > 1:
            ksize = kernel_size
            if stride % 2 == 0:
                ksize = max(kernel_size, stride + 1)
            else:
                ksize = max(kernel_size, stride)
            kernel_size = ksize

        self.loc_cls_layer = conv_fn(
            in_channels,
            n_anchors * (n_coordinates + n_classes),
            kernel_size, 1,
            use_norm=False, use_act=False,
        )

        self.n_coordinates = n_coordinates
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.k_size = kernel_size
        self.stride = stride
        self.in_channel = in_channels

    def _sample_fm(self, x, stride):
        H, W = x.shape[-2:]
        device = x.device
        start = max(0, stride // 2)
        idx_h = torch.arange(start, H, stride, device=device)
        idx_w = torch.arange(start, W, stride, device=device)
        x = torch.index_select(x, dim=-1, index=idx_w)
        x = torch.index_select(x, dim=-2, index=idx_h)
        return x

    def forward(self, x):
        if self.proj_layer is not None:
            x = self.proj_layer(x)
        x = self.loc_cls_layer(x)
        if self.stride > 1:
            x = self._sample_fm(x, self.stride)

        B = x.shape[0]
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(B, -1, self.n_coordinates + self.n_classes)
        loc, cls = torch.split(x, [self.n_coordinates, self.n_classes], dim=-1)
        return loc, cls


class SSDAnchorGenerator:
    def __init__(self, output_strides, n_anchors_list, image_size=(320, 320)):
        self.output_strides = output_strides
        self.n_anchors_list = n_anchors_list
        self.image_size = image_size
        # Default anchor sizes: linearly interpolated between min and max size
        self.min_size = 0.1
        self.max_size = 0.9
        self.aspect_ratios = [0.5, 1.0, 2.0]

    def num_anchors_per_os(self):
        return self.n_anchors_list

    def _compute_anchor_size(self, os_idx):
        n = len(self.output_strides)
        min_s = self.min_size + (self.max_size - self.min_size) * os_idx / n
        max_s = self.min_size + (self.max_size - self.min_size) * (os_idx + 1) / n
        return min_s, max_s

    def __call__(self, fm_height, fm_width, os_idx, device="cpu"):
        stride = self.output_strides[os_idx]
        n_anchors = self.n_anchors_list[os_idx]
        img_h, img_w = self.image_size

        ctr_x = (torch.arange(fm_width, device=device).float() + 0.5) * stride / img_w
        ctr_y = (torch.arange(fm_height, device=device).float() + 0.5) * stride / img_h
        ctr_y, ctr_x = torch.meshgrid(ctr_y, ctr_x, indexing="ij")
        ctr_x = ctr_x.reshape(-1)
        ctr_y = ctr_y.reshape(-1)
        n_pos = ctr_x.numel()

        min_s, max_s = self._compute_anchor_size(os_idx)
        scales = [min_s, (min_s * max_s) ** 0.5]

        # Generate 4 anchor boxes per position (standard SSD)
        ars = [0.5, 1.0, 2.0]
        anchors = []
        for ar in ars[:n_anchors]:
            w = scales[0] * math.sqrt(ar)
            h = scales[0] / math.sqrt(ar)
            anchors.append(torch.stack([ctr_x, ctr_y,
                torch.full_like(ctr_x, w), torch.full_like(ctr_x, h)], dim=1))

        if n_anchors > 3:
            w = scales[1] * math.sqrt(1.0)
            h = scales[1] / math.sqrt(1.0)
            anchors.append(torch.stack([ctr_x, ctr_y,
                torch.full_like(ctr_x, w), torch.full_like(ctr_x, h)], dim=1))

        return torch.cat(anchors, dim=0)


class SSDLoss(nn.Module):
    def __init__(self, n_classes, neg_pos_ratio=3, alpha=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

    def forward(self, predictions, targets):
        loc_pred, conf_pred, anchors = predictions
        batch_size = loc_pred.shape[0]
        device = loc_pred.device
        dtype = loc_pred.dtype

        total_loc_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_conf_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for b in range(batch_size):
            gt_boxes_b = targets[b]["boxes"] if isinstance(targets, list) else targets["boxes"][b]
            gt_labels_b = targets[b]["labels"] if isinstance(targets, list) else targets["labels"][b]
            matched_boxes, matched_labels = self.match_anchors(
                anchors[b], gt_boxes_b, gt_labels_b
            )

            pos_mask = matched_labels > 0
            n_pos = pos_mask.sum().item()

            loc_loss_b = torch.tensor(0.0, device=device, dtype=dtype)
            if n_pos > 0:
                encoded = self.encode_boxes(matched_boxes[pos_mask], anchors[b][pos_mask])
                loc_loss_b = F.smooth_l1_loss(
                    loc_pred[b][pos_mask], encoded, reduction="sum"
                )

            conf_loss_all = F.cross_entropy(
                conf_pred[b].reshape(-1, self.n_classes),
                matched_labels,
                reduction="none"
            )

            n_neg = min(int(n_pos * self.neg_pos_ratio), anchors.shape[1] - n_pos)
            if n_pos > 0 and n_neg > 0:
                neg_loss = conf_loss_all.clone()
                neg_loss[pos_mask] = 0.0
                _, idx = neg_loss.sort(descending=True)
                neg_mask = torch.zeros_like(neg_loss, dtype=torch.bool)
                neg_mask[idx[:n_neg]] = True
                conf_loss_b = conf_loss_all[pos_mask].sum() + conf_loss_all[neg_mask].sum()
            elif n_pos > 0:
                conf_loss_b = conf_loss_all[pos_mask].sum()
            elif n_neg > 0:
                _, idx = conf_loss_all.sort(descending=True)
                conf_loss_b = conf_loss_all[idx[:n_neg]].sum()
            else:
                conf_loss_b = torch.tensor(0.0, device=device, dtype=dtype)

            total_loc_loss = total_loc_loss + loc_loss_b
            total_conf_loss = total_conf_loss + conf_loss_b

        total_loc_loss /= batch_size
        total_conf_loss /= batch_size

        return {
            "loc_loss": total_loc_loss,
            "conf_loss": total_conf_loss,
            "total_loss": total_loc_loss + self.alpha * total_conf_loss,
        }

    def match_anchors(self, anchors, gt_boxes, gt_labels):
        device = anchors.device
        n_anchors = anchors.shape[0]
        n_gt = gt_boxes.shape[0]

        matched_labels = torch.zeros(n_anchors, dtype=torch.long, device=device)
        matched_boxes = anchors.clone()

        if n_gt == 0:
            return matched_boxes, matched_labels

        anchors_xy1 = anchors[:, :2] - anchors[:, 2:] / 2
        anchors_xy2 = anchors[:, :2] + anchors[:, 2:] / 2
        gt_xy1 = gt_boxes[:, :2] - gt_boxes[:, 2:] / 2
        gt_xy2 = gt_boxes[:, :2] + gt_boxes[:, 2:] / 2

        inter_x1 = torch.max(anchors_xy1[:, None, 0], gt_xy1[:, 0])
        inter_y1 = torch.max(anchors_xy1[:, None, 1], gt_xy1[:, 1])
        inter_x2 = torch.min(anchors_xy2[:, None, 0], gt_xy2[:, 0])
        inter_y2 = torch.min(anchors_xy2[:, None, 1], gt_xy2[:, 1])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        anchor_area = anchors[:, 2] * anchors[:, 3]
        gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
        union = anchor_area[:, None] + gt_area - inter
        iou = inter / (union + 1e-8)

        best_gt_iou, best_gt_idx = iou.max(dim=1)

        matched_labels[best_gt_iou >= 0.5] = gt_labels[best_gt_idx[best_gt_iou >= 0.5]]
        matched_boxes[best_gt_iou >= 0.0] = gt_boxes[best_gt_idx[best_gt_iou >= 0.0]]

        best_anchor_iou, best_anchor_idx = iou.max(dim=0)
        for j in range(n_gt):
            if best_anchor_iou[j] > 0:
                idx = best_anchor_idx[j]
                matched_labels[idx] = gt_labels[j]
                matched_boxes[idx] = gt_boxes[j]

        return matched_boxes, matched_labels

    def encode_boxes(self, boxes, anchors):
        eps = 1e-8
        xy = (boxes[:, :2] - anchors[:, :2]) / (anchors[:, 2:] + eps)
        wh = torch.log(boxes[:, 2:] / (anchors[:, 2:] + eps))
        return torch.cat([xy, wh], dim=1)


class GhostMobileViTSSD(nn.Module):
    def __init__(
        self,
        n_classes=80,
        ghost_ratio=0.5,
        use_ghost=False,
        output_strides=None,
        proj_channels=None,
        anchor_sizes=None,
        aspect_ratios=None,
        image_size=(320, 320),
        use_fpn=False,
        fpn_channels=256,
    ):
        super().__init__()

        if output_strides is None:
            output_strides = [16, 32, 64, 128]
        if proj_channels is None:
            proj_channels = [512, 256, 256, 128, 128, 64]
        if anchor_sizes is None:
            anchor_sizes = [
                [0.1, 0.2], [0.2, 0.375], [0.375, 0.55],
                [0.55, 0.725], [0.725, 0.9], [0.9, 1.05]
            ]
        if aspect_ratios is None:
            aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

        self.output_strides = output_strides
        self.image_size = image_size
        self.n_classes = n_classes

        self.encoder = GhostMobileViT(ghost_ratio=ghost_ratio, use_ghost=use_ghost)
        self.encoder.classifier = None
        self.encoder.conv_1x1_exp = None

        self.enc_l3_channels = 96
        self.enc_l4_channels = 128
        self.enc_l5_channels = 160

        extra_layers = {}
        enc_channels = []
        in_ch = self.enc_l5_channels

        n_proj = len(proj_channels)
        for idx, os in enumerate(output_strides):
            out_ch = proj_channels[idx] if idx < n_proj else 256
            if os == 8:
                enc_channels.append(self.enc_l3_channels)
            elif os == 16:
                enc_channels.append(self.enc_l4_channels)
            elif os == 32:
                enc_channels.append(self.enc_l5_channels)
            elif os > 32:
                extra_layers[f"os_{os}"] = SeparableConv(
                    in_ch, out_ch, 3, stride=2,
                    use_norm=True, use_act=True
                )
                enc_channels.append(out_ch)
                in_ch = out_ch
            elif os == -1:
                extra_layers[f"os_{os}"] = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    ConvLayer(in_ch, out_ch, 1, 1, use_norm=True, use_act=False),
                )
                enc_channels.append(out_ch)
                in_ch = out_ch

        self.extra_layers = None
        if extra_layers:
            self.extra_layers = nn.ModuleDict(extra_layers)

        self.fpn = None
        if use_fpn:
            self.fpn = FeaturePyramidNetwork(
                in_channels=enc_channels,
                out_channels=fpn_channels,
            )
            enc_channels = [fpn_channels] * len(output_strides)
            proj_channels = enc_channels

        n_anchors_per_os = [len(ar) + 2 for ar in aspect_ratios[:len(output_strides)]]
        self.anchor_generator = SSDAnchorGenerator(
            output_strides, n_anchors_per_os, image_size
        )
        anchors_ar = self.anchor_generator.num_anchors_per_os()

        self.ssd_heads = nn.ModuleList()
        for os, in_dim, proj_dim, n_anch in zip(
            output_strides, enc_channels, proj_channels, anchors_ar
        ):
            self.ssd_heads.append(SSDHead(
                in_channels=in_dim,
                n_classes=n_classes,
                n_anchors=n_anch,
                proj_channels=proj_dim,
                kernel_size=3 if os != -1 else 1,
                stride=1,
            ))

    def get_backbone_features(self, x):
        endpoints = self.encoder.extract_end_points_all(x)
        out = {}

        for idx, os in enumerate(self.output_strides):
            if os == 8:
                out[f"os_{os}"] = endpoints["out_l3"]
            elif os == 16:
                out[f"os_{os}"] = endpoints["out_l4"]
            elif os == 32:
                out[f"os_{os}"] = endpoints["out_l5"]
            elif os > 32:
                prev_os = self.output_strides[idx - 1]
                out[f"os_{os}"] = self.extra_layers[f"os_{os}"](out[f"os_{prev_os}"])
            elif os == -1:
                prev_os = self.output_strides[idx - 1]
                out[f"os_{os}"] = self.extra_layers[f"os_{os}"](out[f"os_{prev_os}"])

        if self.fpn is not None:
            out = self.fpn(out)

        return out

    def forward(self, x):
        if isinstance(x, dict):
            x = x["image"]

        device = x.device
        B = x.shape[0]

        endpoints = self.get_backbone_features(x)

        locations = []
        confidences = []
        anchors = []

        for idx, (os, head) in enumerate(zip(self.output_strides, self.ssd_heads)):
            fm = endpoints[f"os_{os}"]
            h, w = fm.shape[2:]
            loc, conf = head(fm)
            locations.append(loc)
            confidences.append(conf)

            anchors_fm = self.anchor_generator(h, w, idx, device=device)
            anchors.append(anchors_fm)

        locations = torch.cat(locations, dim=1)
        confidences = torch.cat(confidences, dim=1)
        anchors = torch.cat(anchors, dim=0).unsqueeze(0).expand(B, -1, -1)

        output = {
            "scores": confidences,
            "boxes": locations,
            "anchors": anchors,
        }
        return output

    @torch.no_grad()
    def predict(self, x, conf_threshold=0.01, nms_threshold=0.5, top_k=400,
                objects_per_image=200):
        self.eval()
        orig_dtype = x.dtype
        device = x.device

        output = self.forward(x)
        scores = F.softmax(output["scores"], dim=-1)
        anchors = self.anchor_generator(
            *[s // os for s, os in zip(self.image_size, self.output_strides)],
            self.output_strides,
            device=device,
            image_size=self.image_size,
        )
        results = self.decode_predictions(
            output["boxes"], output["scores"], anchors.unsqueeze(0),
            conf_threshold, nms_threshold, top_k, objects_per_image
        )
        return results

    def decode_predictions(self, loc_pred, conf_pred, anchors,
                           conf_threshold, nms_threshold, top_k,
                           objects_per_image):
        batch_size = loc_pred.shape[0]
        n_classes = conf_pred.shape[-1]
        device = loc_pred.device

        results = []
        for b in range(batch_size):
            boxes = self.decode_boxes(loc_pred[b], anchors[b])
            scores = F.softmax(conf_pred[b], dim=-1)

            det_boxes, det_scores, det_labels = [], [], []

            for c in range(1, n_classes):
                mask = scores[:, c] > conf_threshold
                cls_scores = scores[mask, c]
                if cls_scores.size(0) == 0:
                    continue
                cls_boxes = boxes[mask]

                num = min(top_k, cls_scores.size(0))
                cls_scores, idx = cls_scores.topk(num)
                cls_boxes = cls_boxes[idx]

                det_scores.append(cls_scores)
                det_boxes.append(cls_boxes)
                det_labels.append(torch.full_like(
                    cls_scores, c, dtype=torch.long, device=device
                ))

            if not det_scores:
                results.append({
                    "boxes": torch.empty(0, 4, device=device),
                    "scores": torch.empty(0, device=device),
                    "labels": torch.empty(0, dtype=torch.long, device=device),
                })
                continue

            det_scores = torch.cat(det_scores)
            det_boxes = torch.cat(det_boxes)
            det_labels = torch.cat(det_labels)

            keep = torchvision.ops.batched_nms(
                det_boxes, det_scores, det_labels, nms_threshold
            )
            keep = keep[:objects_per_image]

            results.append({
                "boxes": det_boxes[keep],
                "scores": det_scores[keep],
                "labels": det_labels[keep],
            })
        return results

    def decode_boxes(self, loc, anchors):
        xy = loc[:, :2] * anchors[:, 2:] + anchors[:, :2]
        wh = torch.exp(loc[:, 2:]) * anchors[:, 2:]
        return torch.cat([xy - wh / 2, xy + wh / 2], dim=1)


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for i, in_ch in enumerate(in_channels):
            lateral = ConvLayer(in_ch, out_channels, 1, 1, use_norm=True, use_act=False)
            output = ConvLayer(out_channels, out_channels, 3, 1, use_norm=True, use_act=True)
            self.lateral_convs.append(lateral)
            self.output_convs.append(output)

        self.out_channels = out_channels

    def forward(self, inputs):
        names = sorted(inputs.keys())
        laterals = [
            lateral(inputs[name])
            for lateral, name in zip(self.lateral_convs, names)
        ]

        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode="nearest"
            )

        outputs = {}
        for i, name in enumerate(names):
            outputs[name] = self.output_convs[i](laterals[i])
        return outputs


import torchvision
