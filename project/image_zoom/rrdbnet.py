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
import os
import torch
from torch import nn as nn
from torch.nn import functional as F
import pdb

# def pixel_unshuffle(x, scale: int):
#     """Pixel unshuffle."""

#     b, c, hh, hw = x.size()
#     out_channel = int(c * (scale ** 2))
#     assert hh % scale == 0 and hw % scale == 0
#     h = hh // scale
#     w = hw // scale
#     x_view = x.view(b, c, h, scale, w, scale)
#     x_view = x_view.permute(0, 1, 3, 5, 2, 4) # b, (c * scale * scale), h, w

#     return x_view.reshape(b, out_channel, h, w)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        #del x1, x2, x3, x4

        # Empirically, use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
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
    def __init__(self, model_name, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        if scale == 4:
            self.MAX_H = 640
            self.MAX_W = 1024
        else: # scale == 2
            self.MAX_H = 1024
            self.MAX_W = 2048
        self.MAX_TIMES = scale
        #  GPU memory
        #  zoom4x -- 640x1024, 8.8G, 1000ms;
        #  zoom2x -- 1024x2048, 7.4G, 2600ms
        #  anime4x -- 512x1024, 7.4G, 800ms

        if scale == 2:
            num_in_ch = num_in_ch * 4
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if scale == 2:
            self.unshffle = nn.PixelUnshuffle(scale)
        else:
            self.unshffle = nn.Identity()

        self.load_weights(model_path=f"models/{model_name}.pth")
        # self.half()

    # def on_cuda(self):
    #     return self.conv_first.weight.is_cuda # model is on cuda ? 

    def load_weights(self, model_path="models/image_zoom4x.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        sd = torch.load(checkpoint)
        self.load_state_dict(sd['params_ema'])


    def forward(self, x):
        B, C, H, W = x.size()

        pad_h = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
        pad_w = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')

        feat = self.unshffle(x)
        feat = self.conv_first(feat)

        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="bicubic", align_corners=True)))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="bicubic", align_corners=True)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out[:, :, 0:H*self.MAX_TIMES, 0:W*self.MAX_TIMES].clamp(0.0, 1.0)


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.
    """
    def __init__(self, model_name, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 4

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = float(upscale)
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

        self.load_weights(model_path=f"models/{model_name}.pth")
        # self.half()

    def load_weights(self, model_path="models/video_anime4x.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        sd = torch.load(checkpoint)
        self.load_state_dict(sd['params'])

    # def on_cuda(self):
    #     return self.body[0].weight.is_cuda # model is on cuda ? 

    def forward(self, x):
        out = x
        # for i in range(0, len(self.body)):
        #     out = self.body[i](out)
        for i, layer in enumerate(self.body):
            out = layer(out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


class SRVGGNetDenoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 4

        self.model1 = SRVGGNetCompact("image_denoise1", num_conv=32)
        self.model2 = SRVGGNetCompact("image_denoise2", num_conv=32)

        # self.half()

    def on_cuda(self):
        return self.model1.body[0].weight.is_cuda # model is on cuda ? 

    def forward(self, x, noise_strength: float=0.0):
        out = x
        # for i in range(0, len(self.body)):
        #     out = self.body[i](out)
        for layer1, layer2 in zip(self.model1.body, self.model2.body):
            out = noise_strength * layer1(out) + (1.0 - noise_strength) * layer2(out)

        out = noise_strength * self.model1.upsampler(out) + (1.0 - noise_strength) * self.model2.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.model1.upscale, mode='nearest')
        out += base
        return out

