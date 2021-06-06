from torch import nn as nn
from mmdet.models import BACKBONES
import time
import numpy as np
import torch
from torch import nn
from ..model_utils import ModulatedDeformConvBlock
from torchplus.tools import change_default_args
from torchplus.nn.modules.common import Sequential
from mmcv.runner import load_checkpoint
from mmcv.cnn import build_norm_layer
from torchvision.models import resnet
from mmcv.cnn import (build_norm_layer, build_upsample_layer, constant_init,
                      is_norm, kaiming_init)

@BACKBONES.register_module()
class SECONDFPNDCN(nn.Module):
    def __init__(self,
                 in_channels=128,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 out_channels=(256, 256, 256),
                 use_dcn=True):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(SECONDFPNDCN, self).__init__()
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.layer_nums = layer_nums
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = out_channels
        self.num_input_features = in_channels
        self.use_dcn=use_dcn


        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(out_channels) == len(upsample_strides)
        self.upsample_start_idx = len(layer_nums) - len(upsample_strides)

        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self.upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        BatchNorm2d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        self.activation_fcn=change_default_args(negative_slope=0.01,inplace=True)(nn.LeakyReLU)

        in_filters = [in_channels, *num_filters[:-1]]

        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self.upsample_start_idx >= 0:
                stride = upsample_strides[i - self.upsample_start_idx]
                stride = np.round(stride).astype(np.int64)
                if self.use_dcn:
                    deblock=Sequential(ModulatedDeformConvBlock(num_out_filters,num_out_filters,act_fn=self.activation_fcn))
                    deblock.add(ConvTranspose2d(num_out_filters,
                            self.num_upsample_filters[i - self.upsample_start_idx],
                            stride,
                            stride=stride),)
                    deblock.add(BatchNorm2d(self.num_upsample_filters[i -self.upsample_start_idx]))
                    deblock.add(self.activation_fcn())
                else:
                    deblock=Sequential(ConvTranspose2d(num_out_filters,
                            self.num_upsample_filters[i - self.upsample_start_idx],
                            stride,
                            stride=stride),
                            BatchNorm2d(self.num_upsample_filters[i -self.upsample_start_idx]),
                            self.activation_fcn())

                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)




    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)

        block = Sequential(
            Conv2d(inplanes, planes, 3, padding=1,stride=stride),
            BatchNorm2d(planes),
            self.activation_fcn())

        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1,dilation=1))
            block.add(BatchNorm2d(planes))
            block.add(self.activation_fcn())

        return block, planes



    @property
    def downsample_factor(self):
        factor = np.prod(self.layer_strides)
        if len(self.upsample_strides) > 0:
            factor /= self.upsample_strides[-1]
        return factor

    def forward(self, x):
        ups = []
        stage_outputs = []
        out=[]
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self.upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self.upsample_start_idx](x))
                out.append(ups[-1])
            else:
                out.append(stage_outputs[-1])


        x = [torch.cat(out, dim=1)]


        return x

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         kaiming_init(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         constant_init(m, 1)

        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

