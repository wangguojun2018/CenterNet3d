import torch
from torch.nn import functional as F
from torch import nn
from mmdet.models import HEADS
from collections import defaultdict
from mmcv.cnn import bias_init_with_prob, normal_init
import torch
from torch import nn as nn

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models import HEADS
from ..builder import build_loss
from mmdet.models.losses import MSELoss
from mmdet3d.models.losses import ModifiedFocalLoss

@HEADS.register_module()
class Center3DHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 train_cfg,
                 test_cfg,
                 bbox_coder=dict(type="XYZWLHRBoxCoder",voxel_size=[0.05,0.05,0.1],
                                pc_range=[0, -40, -3, 70.4, 40, 1],num_dir_bins=12,
                                downsample_ratio=4.0,min_overlap=0.01),
                 loss_cls=dict(
                     type='ModifiedFocalLoss',loss_weight=0.5),
                 loss_xy=dict(
                     type='GatherBalancedL1Loss',loss_weight=1.0),
                 loss_z=dict(
                     type='GatherBalancedL1Loss', loss_weight=1.0),
                 loss_dim=dict(
                     type='GatherBalancedL1Loss', loss_weight=1.0),
                 loss_dir=dict(
                     type='GatherBinResLoss', loss_weight=1.0),
                 bias_cls=None,
                 loss_corner=None,
                 loss_decode=None,
                 ):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(Center3DHead, self).__init__()
        self.num_class = num_classes
        self.in_channels=in_channels
        self.tran_cfg=train_cfg
        self.test_cfg=test_cfg
        self.corner_attention=False
        if loss_corner is not None:
            self.corner_attention=True

        self.activaton_fun = nn.LeakyReLU(0.01, inplace=True)

        #build box coder
        self.box_coder=build_bbox_coder(bbox_coder)

        self.loss_cls=build_loss(loss_cls)
        self.loss_xy = build_loss(loss_xy)
        self.loss_z = build_loss(loss_z)
        self.loss_dim = build_loss(loss_dim)
        if loss_dir['type']=="GatherBinResLoss":
            loss_dir['num_dir_bins']=bbox_coder['num_dir_bins']
        self.loss_dir = build_loss(loss_dir)

        self.loss_decode=self.loss_corner=None
        if loss_corner is not None:
            self.loss_corner=build_loss(loss_corner)
            print("use corner attention module!")
        if loss_decode is not None:
            loss_decode["box_coder"]=bbox_coder
            self.loss_decode=build_loss(loss_decode)
            print("use decode loss!")

        if bias_cls is None:
            bias_cls = bias_init_with_prob(0.01)

        self.heads = {"center_pred": 1,"xy_pred": 2, "z_pred": 1,"dim_pred": 3,  "dir_pred": 2}
        if bbox_coder["num_dir_bins"]>0:
            assert loss_dir["type"]=="GatherBinResLoss","num_dir_bins greater than 0, GatherBinResLoss is required"
            self.heads["dir_pred"]=bbox_coder["num_dir_bins"]*2

        if self.corner_attention:
            self.heads['corner_pred'] = 1

        for head in self.heads:
            classes = self.heads[head]
            # if head in ["dim_pred"]:
            #     fc = nn.Sequential(
            #         nn.Conv2d(in_channels, feat_channels,
            #                   kernel_size=3, padding=1, bias=True),
            #         self.activaton_fun,
            #         nn.Conv2d(feat_channels, classes,
            #                   kernel_size=3, stride=1,padding=1))
            # else:
            fc = nn.Sequential(
                nn.Conv2d(in_channels, feat_channels,
                          kernel_size=3, padding=1, bias=True),
                self.activaton_fun,
                nn.Conv2d(feat_channels, classes,
                          kernel_size=1, stride=1, ))
            if head in ["center_pred","corner_pred"]:
                fc[-1].bias.data.fill_(bias_cls)
            self.__setattr__(head, fc)

    def forward(self, x):
        z = {}
        x=x[0]
        # if not self.training and self.corner_attention:
        #     self.heads.pop('corner_pred')

        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
            if head in ["center_pred","corner_pred","xy_pred"]:
                z[head]=torch.sigmoid(z[head])
            if head in ["dir_pred"] and self.heads["dir_pred"]==2:
                z[head]=torch.tanh(z[head])
        return z

    def init_weights(self):
        """Initialize the weights of head."""
        # bias_cls = bias_init_with_prob(0.01)
        # normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        # normal_init(self.conv_reg, std=0.01)
        pass

    def get_bboxes(self,pred_dicts,input_metas):
        return self.box_coder.decode_center(pred_dicts,input_metas,score_threshold=self.test_cfg['score_thr'])

    def loss(self,pred_dict,gt_labels,gt_bboxes,img_metas=None):
        gt_dict=self.box_coder.generate_target(gt_labels,gt_bboxes)
        mask=gt_dict["reg_mask"]
        index=gt_dict["gt_index"]

        if isinstance(self.loss_cls, MSELoss):
            avg_fac=gt_dict["score_map"].sum()
            cls_loss = self.loss_cls(pred_dict["center_pred"], gt_dict["score_map"],avg_factor=avg_fac)
        elif isinstance(self.loss_cls, ModifiedFocalLoss):
            cls_loss=self.loss_cls(pred_dict["center_pred"],gt_dict["score_map"])
        else:
            raise NotImplementedError

        xy_loss=self.loss_xy(pred_dict["xy_pred"],mask,index,gt_dict["gt_xyz"][...,:2])
        z_loss=self.loss_z(pred_dict["z_pred"],mask,index,gt_dict["gt_xyz"][...,2:])
        dim_loss=self.loss_dim(pred_dict["dim_pred"],mask,index,gt_dict["gt_dim"])
        dir_loss=self.loss_dir(pred_dict["dir_pred"],mask,index,gt_dict["gt_dir"])
        # total_loss = cls_loss + xy_loss + z_loss + dim_loss + dir_loss
        loss_dict = {
            "cls_loss": cls_loss,
            "xy_loss": xy_loss,
            "z_loss": z_loss,
            "dim_loss": dim_loss,
            "dir_loss": dir_loss,
        }
        if self.loss_corner is not None:
            if isinstance(self.loss_corner, MSELoss):
                avg_fac = gt_dict["corner_map"].sum()
                corner_loss = self.loss_corner(pred_dict["corner_pred"], gt_dict["corner_map"], avg_factor=avg_fac)
            elif isinstance(self.loss_corner, ModifiedFocalLoss):
                corner_loss = self.loss_corner(pred_dict["corner_pred"], gt_dict["corner_map"])
            else:
                raise NotImplementedError
            # corner_loss=self.loss_corner(pred_dict["corner_pred"],gt_dict["corner_map"])
            loss_dict["corner_loss"]=corner_loss

        if self.loss_decode is not None:
            decode_loss=self.loss_decode(pred_dict,mask,index,gt_dict["gt_boxes3d"],gt_dict["gt_dir"])
            loss_dict["decode_loss"]=decode_loss

        return loss_dict