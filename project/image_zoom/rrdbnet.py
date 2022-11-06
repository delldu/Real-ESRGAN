"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 13日 星期二 00:22:40 CST
# ***
# ************************************************************************************/
#
import torch
from torch import nn as nn
from torch.nn import functional as F


def pixel_unshuffle(x, scale: int):
    """Pixel unshuffle."""
    b, c, hh, hw = x.size()
    out_channel = int(c * (scale ** 2))
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN.
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        # Define max GPU/CPU memory -- 7G(2x: 1024x1024, 4x out of cuda memory !!!)
        # 4X: 800x800 --> 9G, 720x720 --> 8G
        self.MAX_H = 800
        self.MAX_W = 800
        self.MAX_TIMES = 8

        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward_x(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="bicubic", align_corners=True)))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="bicubic", align_corners=True)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out.clamp(0.0, 1.0)

    def forward(self, x):
        # Need Resize ?
        B, C, H, W = x.size()
        if H > self.MAX_H or W > self.MAX_W:
            s = min(self.MAX_H / H, self.MAX_W / W)
            SH, SW = int(s * H), int(s * W)
            resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)
        else:
            resize_x = x

        # Need Pad ?
        PH, PW = resize_x.size(2), resize_x.size(3)
        if PH % self.MAX_TIMES != 0 or PW % self.MAX_TIMES != 0:
            r_pad = self.MAX_TIMES - (PW % self.MAX_TIMES)
            b_pad = self.MAX_TIMES - (PH % self.MAX_TIMES)
            resize_pad_x = F.pad(resize_x, (0, r_pad, 0, b_pad), mode="replicate")
        else:
            resize_pad_x = resize_x

        y = self.forward_x(resize_pad_x)

        if self.scale == 4:
            y = y[:, :, 0 : 4 * PH, 0 : 4 * PW]  # Remove Zero Pads, 4 -- zoom 4x
            y = F.interpolate(y, size=(4 * H, 4 * W), mode="bilinear", align_corners=False)
        else:  # Zoom2x
            y = y[:, :, 0 : 2 * PH, 0 : 2 * PW]  # Remove Zero Pads, 2 -- zoom 2x
            y = F.interpolate(y, size=(2 * H, 2 * W), mode="bilinear", align_corners=False)

        return y
