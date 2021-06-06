#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from mmdet3d.core.bbox.coders import gather_feature
from mmdet.models.builder import LOSSES
from mmdet.core import build_bbox_coder

@LOSSES.register_module()
class ModifiedFocalLoss(nn.Module):
    def __init__(self,loss_weight,reduction="mean"):
        super(ModifiedFocalLoss,self).__init__()
        self.weight=loss_weight
        self.reduction=reduction

    def forward(self,pred,target):
        loss=modified_focal_loss(pred,target,reduction=self.reduction)
        loss=loss*self.weight
        return loss

@LOSSES.register_module()
class GatherBalancedL1Loss(nn.Module):
    def __init__(self,loss_weight,beta=1.0, alpha=0.5, gamma=1.5,reduction="none"):
        super(GatherBalancedL1Loss,self).__init__()
        self.beta=beta
        self.alpha=alpha
        self.gamma=gamma
        self.weight=loss_weight
        self.reduction=reduction
        assert reduction=="none","only none reduction is support!"

    def forward(self,output,mask,index,target):
        pred = gather_feature(output, index, use_transform=True)  # (-1,C)
        mask = mask.unsqueeze(dim=2).expand_as(pred).float()
        pred=pred*mask
        target=target*mask

        assert pred.size() == target.size() and target.numel() > 0
        loss=balanced_l1_loss(pred,target,beta=self.beta,alpha=self.alpha,gamma=self.gamma,reduction=self.reduction)
        loss = loss.sum() / (mask.sum() + 1e-4)*self.weight
        return loss

@LOSSES.register_module()
class BalancedL1LossV2(nn.Module):
    def __init__(self,loss_weight,beta=1.0, alpha=0.5, gamma=1.5,reduction="none"):
        super(BalancedL1LossV2,self).__init__()
        self.beta=beta
        self.alpha=alpha
        self.gamma=gamma
        self.weight=loss_weight
        self.reduction=reduction
        assert reduction=="none","only none reduction is support!"

    def forward(self,output,target,mask):

        assert output.size() == target.size()
        mask = mask.unsqueeze(dim=1).expand_as(output).float()
        loss=balanced_l1_loss(output,target,beta=self.beta,alpha=self.alpha,gamma=self.gamma,reduction=self.reduction)
        loss=loss*mask
        loss = loss.sum() / (mask.sum() + 1e-4)*self.weight
        return loss

@LOSSES.register_module()
class GatherL1Loss(nn.Module):
    def __init__(self,loss_weight,reduction="none"):
        super(GatherL1Loss,self).__init__()
        self.weight=loss_weight
        self.reduction=reduction
        assert reduction=="none","only none reduction is support!"
    def forward(self,output,mask,index,target):
        pred = gather_feature(output, index, use_transform=True)  # (-1,C)
        mask = mask.unsqueeze(dim=2).expand_as(pred).float()
        pred=pred*mask
        target=target*mask
        assert pred.size() == target.size() and target.numel() > 0
        loss = F.l1_loss(pred * mask, target * mask, reduction=self.reduction)
        loss = loss / (mask.sum() + 1e-4)*self.weight
        return loss

@LOSSES.register_module()
class GatherBinResLoss(nn.Module):
    def __init__(self,loss_weight,num_dir_bins=12,reduction="none"):
        super(GatherBinResLoss,self).__init__()
        self.weight=loss_weight
        self.reduction=reduction
        self.num_dir_bins=num_dir_bins

    def dir_bin_res_loss(self,dir_preds,mask,index,gt_dir):
        preds = gather_feature(dir_preds, index, use_transform=True)  # (B,-1,C)

        pred_bin=preds[...,:self.num_dir_bins]
        pred_reg=preds[...,self.num_dir_bins:]

        gt_bin=gt_dir[...,0]
        gt_bin=gt_bin.long()
        gt_reg=gt_dir[...,1]
        mask = mask.float()
        ry_bin_onehot = gt_bin.new_zeros(gt_bin.size(0),gt_bin.size(1),self.num_dir_bins)
        ry_bin_onehot.scatter_(2, gt_bin.unsqueeze(-1), 1)
        loss_ry_bin = F.cross_entropy(pred_bin.view(-1,pred_bin.size(-1)),
                                      gt_bin.view(-1),reduction='none')
        loss_ry_res = F.smooth_l1_loss((pred_reg *ry_bin_onehot).sum(dim=-1),
                                       gt_reg,reduction='none')
        loss_ry_res = (loss_ry_res * mask).sum() / (mask.sum() + 1e-4)
        loss_ry_bin = (loss_ry_bin * mask.reshape(-1)).sum() / (mask.reshape(-1).sum() + 1e-4)
        return loss_ry_bin+loss_ry_res


    def forward(self,dir_preds,mask,index,gt_dir):
        loss=self.dir_bin_res_loss(dir_preds,mask,index,gt_dir)
        return loss*self.weight


@LOSSES.register_module()
class BinResLoss(nn.Module):
    def __init__(self,loss_weight,num_rad_bin=12,reduction="none"):
        super(BinResLoss,self).__init__()
        self.weight=loss_weight
        self.reduction=reduction
        self.num_rad_bin=num_rad_bin

    def dir_bin_res_loss(self,dir_preds,gt_dir,mask):


        pred_bin=dir_preds[:,:self.num_rad_bin,:,:]
        pred_reg=dir_preds[:,self.num_rad_bin:,:,:]

        gt_bin=gt_dir[:,0:1,:,:]
        gt_bin=gt_bin.long()
        gt_reg=gt_dir[:,1:2,:,:]
        # mask = mask.unsqueeze(1).expand_as(pred_bin).float()
        ry_bin_onehot = torch.cuda.FloatTensor(pred_bin.size(0),pred_bin.size(1),pred_bin.size(2),pred_bin.size(3)).zero_()
        ry_bin_onehot.scatter_(1, gt_bin, 1)
        loss_ry_bin = F.cross_entropy(pred_bin,
                                      gt_bin.squeeze(1),reduction='none')
        loss_ry_res = F.smooth_l1_loss((pred_reg * ry_bin_onehot).sum(dim=1),
                                       gt_reg.squeeze(1),reduction='none')
        loss_ry_res = (loss_ry_res*mask).sum() / (mask.sum() + 1e-4)
        loss_ry_bin = (loss_ry_bin*mask).sum() / (mask.sum() + 1e-4)
        return loss_ry_bin+loss_ry_res


    def forward(self,dir_preds,gt_dir,mask):
        loss=self.dir_bin_res_loss(dir_preds,gt_dir,mask)
        return loss*self.weight

@LOSSES.register_module()
class Boxes3dDecodeLoss(nn.Module):
    def __init__(self,loss_weight,box_coder=None,beta=1.0, alpha=0.5, gamma=1.5):
        super(Boxes3dDecodeLoss,self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.weight = loss_weight
        self.box_coder = build_bbox_coder(box_coder)

    def forward(self,pred_dict, mask, index, target,gt_dir=None):
        #

        # fmap=example['score_map']
        # print("fmap shape is ",fmap.shape, fmap.dtype,fmap.device)
        voxel_size = self.box_coder.voxel_size
        pc_range = self.box_coder.pc_range
        fmap = pred_dict['center_pred']
        batch,channels,height,width=fmap.shape
        dim_pred = pred_dict['dim_pred']
        dim_pred = gather_feature(dim_pred, index, use_transform=True)
        xy_pred = pred_dict['xy_pred']
        xy_pred = gather_feature(xy_pred, index, use_transform=True)
        z_pred = pred_dict['z_pred']
        z_pred = gather_feature(z_pred, index, use_transform=True)
        dir_pred = pred_dict['dir_pred']
        dir_pred = gather_feature(dir_pred, index, use_transform=True)

        if self.box_coder.num_dir_bins<=0:
            dir_pred=torch.atan2(dir_pred[:, :, 0:1], dir_pred[:, :, 1:])
        else:
            dir_bin =gt_dir[...,0:1].long()
            dir_res= torch.gather(dir_pred[:, :, self.box_coder.num_dir_bins:], dim=-1,
                                        index=dir_bin)
            dir_pred=self.box_coder.class2angle(dir_bin,dir_res)

        ys = (index / width).int().float().unsqueeze(-1)
        xs = (index % width).int().float().unsqueeze(-1)
        xs = xs + xy_pred[:, :, 0:1]
        ys = ys + xy_pred[:, :, 1:2]
        xs = xs * self.box_coder.downsample_ratio * voxel_size[0] + pc_range[0]
        ys = ys * self.box_coder.downsample_ratio * voxel_size[1] + pc_range[1]

        boxes_pred=torch.cat([xs,ys,z_pred,dim_pred,dir_pred],dim=-1).reshape(-1,7)
        boxes_pred_instances=LiDARInstance3DBoxes(boxes_pred,origin=(0.5,0.5,0))
        corners_pred=boxes_pred_instances.corners.reshape(batch,-1,8,3)
        boxes_gt=target.reshape(-1,7)
        boxes_gt_instances=LiDARInstance3DBoxes(boxes_gt,origin=(0.5,0.5,0))
        corners_gt = boxes_gt_instances.corners.reshape(batch,-1,8,3)
        if self.box_coder.num_dir_bins<=0:
            boxes_gt_flip=boxes_gt.clone()
            boxes_gt_flip[:,6]+=np.pi
            boxes_gt_flip_instances=LiDARInstance3DBoxes(boxes_gt_flip,origin=(0.5,0.5,0))
            corners_gt_flip=boxes_gt_flip_instances.corners.reshape(batch,-1,8,3)
            diff= torch.min(torch.abs(corners_pred - corners_gt),torch.abs(corners_pred-corners_gt_flip))
            b = np.e ** (self.gamma / self.alpha) - 1
            loss = torch.where(
                diff < self.beta,
                self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff,
                self.gamma * diff + self.gamma / b - self.alpha * self.beta)
        else:
            loss = balanced_l1_loss(corners_pred, corners_gt, beta=self.beta, alpha=self.alpha, gamma=self.gamma)
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(corners_gt).float()
        loss=loss*mask
        loss = loss.sum() / (mask.sum() + 1e-4)*self.weight
        return loss

def modified_focal_loss(pred, gt,reduction="sum"):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if reduction=="none":
        loss=pos_loss+neg_loss
    elif reduction=="sum":
        loss=pos_loss.sum()+neg_loss.sum()

    elif reduction=="mean":
        num_pos  = pos_inds.float().sum()
        if num_pos == 0:
            loss = neg_loss.sum()
        else:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
    else:
        raise NotImplementedError
    return loss


def balanced_l1_loss(pred,target, beta=1.0, alpha=0.5, gamma=1.5,reduction="none"):

    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta,)
    if reduction=="none":
        loss=loss
    elif reduction=="sum":
        loss=loss.sum()
    elif reduction=="mean":
        loss=loss.mean()
    else:
        raise NotImplementedError
    return loss

def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor


