#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from mmcv.ops import ModulatedDeformConv2d as ModulatedDeformConv2d
from mmcv.ops import DeformConv2d as DeformConv2d
from mmcv.cnn import build_norm_layer

class DeformConvBlock(nn.Module):
    def __init__(self, chi, cho,activation="relu"):
        super(DeformConvBlock, self).__init__()

        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho,eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        if activation=="lrelu":
            self.actf = nn.Sequential(
                nn.BatchNorm2d(cho, eps=1e-3, momentum=0.01),
                nn.LeakyReLU(0.1,inplace=True))
        self.conv = DeformConvWithOff(chi, cho,
            kernel_size=3, deformable_groups=1,)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class ModulatedDeformConvBlock(nn.Module):
    def __init__(self, chi, cho,act_fn):
        super(ModulatedDeformConvBlock, self).__init__()
        self.norm_cfg= dict(type='BN2d', eps=1e-3, momentum=0.01)
        self.actf = nn.Sequential(
            build_norm_layer(self.norm_cfg,cho)[1],
            act_fn()
        )
        self.conv = ModulatedDeformConvWithOff(chi, cho,
            kernel_size=3, deformable_groups=1,)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x



class DeformConvWithOff(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
        super(DeformConvWithOff, self).__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcn = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
        )

    def forward(self, input):
        offset = self.offset_conv(input)
        output = self.dcn(input, offset)
        return output


class ModulatedDeformConvWithOff(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
        super(ModulatedDeformConvWithOff, self).__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcnv2 = ModulatedDeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
        )

        self.init_offset()

    def init_offset(self):
        self.offset_mask_conv.weight.data.zero_()
        self.offset_mask_conv.bias.data.zero_()

    def forward(self, input):
        x = self.offset_mask_conv(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output
