import time

import numpy as np
import spconv
import torch
from torch import nn
from torchplus.tools import change_default_args
from mmdet3d.models.registry import MIDDLE_ENCODERS
from .sparse_encoder_aux import single_conv,double_conv,stride_conv,triple_conv
from torchplus.nn.modules.common import Sequential

BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
BatchNorm2d = change_default_args(
    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
Conv2d = change_default_args(bias=False)(nn.Conv2d)


@MIDDLE_ENCODERS.register_module()
class SparseEncoderV2(nn.Module):
    def __init__(self,sparse_shape,in_channels=128,out_channels=128,name='SparseEncoderV2'):
        super(SparseEncoderV2, self).__init__()
        self.name = name


        print("input sparse shape is ", sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = sparse_shape

        self.activation_fcn=change_default_args(negative_slope=0.01,inplace=True)(nn.LeakyReLU)
        # input: # [1600, 1200, 40]
        self.conv0 = double_conv(in_channels, 16, 'subm0', activation=self.activation_fcn)
        self.down0 = stride_conv(16, 32, 'down0', activation=self.activation_fcn)

        self.conv1 = double_conv(32, 32, 'subm1', activation=self.activation_fcn)  # [20,800,704]
        self.down1 = stride_conv(32, 64, 'down1', activation=self.activation_fcn)

        self.conv2 = triple_conv(64, 64, 'subm2', activation=self.activation_fcn)  # [10,400,352]
        self.down2 = stride_conv(64, 64, 'down2', activation=self.activation_fcn)

        self.conv3 = triple_conv(64, 64, 'subm3', activation=self.activation_fcn)  # [5,200,176]

        self.down3=spconv.SparseSequential(
            SpConv3d(64, 64, (3,1,1), (2,1,1), indice_key="down3"),
            BatchNorm1d(64),
            self.activation_fcn())                                           # [5,200,176]

        # self.down2extra = spconv.SparseSequential(
        #     SpConv3d(64, 64, (3, 1, 1), (2, 1, 1), indice_key="down2extra"),
        #     BatchNorm1d(64),
        #     self.activation_fcn())
        #
        # self.conv2d= Sequential(
        #     Conv2d(256, 128, 3, padding=1,stride=1),
        #     BatchNorm2d(128),
        #     self.activation_fcn())


    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        x= spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # t = time.time()
        x0=self.conv0(x)
        x0=self.down0(x0)

        x1=self.conv1(x0)
        x1=self.down1(x1)
        x2=self.conv2(x1)
        # xconv2=spconv.SparseConvTensor(x2.features.clone(),x2.indices,x2.spatial_shape,x2.batch_size)
        # xconv2.indice_dict=x2.indice_dict
        # xconv2.grid=x2.grid
        # xconv2=self.down2extra(xconv2)
        # xconv2=xconv2.dense()
        # N1,C1,D1,H1,W1=xconv2.shape
        # xconv2=xconv2.view(N1,C1*D1,H1,W1)
        # xconv2=self.conv2d(xconv2)

        x2=self.down2(x2)
        x3=self.conv3(x2)
        x3=self.down3(x3)

        ret = x3.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        # print("ret shape",ret.shape)
        # print("xconv2",xconv2.shape)
        return ret

