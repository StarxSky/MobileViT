import torch
import torch.nn as nn


class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 ratio=0.5, dw_size=3, use_norm=True, use_act=True):
        super().__init__()
        primary_out = max(1, int(out_channels * ratio))
        cheap_out = out_channels - primary_out

        padding = (kernel_size - 1) // 2 if isinstance(kernel_size, int) else \
            tuple((k - 1) // 2 for k in kernel_size)

        self.block = nn.Sequential()
        self.block.add_module("conv", nn.Conv2d(
            in_channels, primary_out, kernel_size, stride,
            padding=padding, bias=False
        ))
        if use_norm:
            self.block.add_module("norm", nn.BatchNorm2d(primary_out))
        if use_act:
            self.block.add_module("act", nn.SiLU(inplace=True))

        self.cheap = None
        if cheap_out > 0:
            dw_padding = (dw_size - 1) // 2
            cheap_seq = nn.Sequential()
            cheap_seq.add_module("conv", nn.Conv2d(
                primary_out, cheap_out, dw_size, 1,
                padding=dw_padding, groups=primary_out, bias=False
            ))
            if use_norm:
                cheap_seq.add_module("norm", nn.BatchNorm2d(cheap_out))
            if use_act:
                cheap_seq.add_module("act", nn.SiLU(inplace=True))
            self.cheap = cheap_seq

    def forward(self, x):
        x = self.block(x)
        if self.cheap is not None:
            x = torch.cat([x, self.cheap(x)], dim=1)
        return x
