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
from mmdet.core import  multiclass_nms,bbox2distance,distance2bbox
from mmdet.models import HEADS
from ..builder import build_loss
from mmdet.models.losses import MSELoss
from mmdet3d.models.losses import ModifiedFocalLoss
from mmcv.cnn import Scale, normal_init
import matplotlib.pyplot as plt


INF = 1e8

@HEADS.register_module()
class Points2DClassificationHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 train_cfg,
                 test_cfg,
                 stride=8,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 bias_cls=None,
                 center_radius=1.5,
                 center_sampling=True,
                 ):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(Points2DClassificationHead, self).__init__()
        self.num_class = num_classes
        self.in_channels=in_channels
        self.tran_cfg=train_cfg
        self.test_cfg=test_cfg
        self.stride=stride
        self.center_radius=center_radius
        self.center_sampling = center_sampling
        self.activaton_fun = nn.LeakyReLU(0.1, inplace=True)

        self.loss_cls=build_loss(loss_cls)

        if bias_cls is None:
            bias_cls = bias_init_with_prob(0.01)
        self.heads = {"cls_preds": 1}

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(in_channels, feat_channels,
                          kernel_size=3, padding=1, bias=True),
                self.activaton_fun,
                nn.Conv2d(feat_channels, classes,
                          kernel_size=1, stride=1, ))
            if head in ["cls_preds"]:
                fc[-1].bias.data.fill_(bias_cls)
            self.__setattr__(head, fc)

    def forward(self, x):
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x[0])

        return z

    def init_weights(self):
        """Initialize the weights of head."""
        # bias_cls = bias_init_with_prob(0.01)
        # normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        # normal_init(self.conv_reg, std=0.01)
        pass

    def get_bboxes(self,
                   pred_dict,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (Tensor): Box scores (N,  num_classes, H, W)
            bbox_preds (list[Tensor]): (N, 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N,  1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # assert len(cls_scores) == len(bbox_preds)
        # num_levels = len(cls_scores)

        cls_scores = pred_dict["cls_preds"]
        bbox_preds = pred_dict["reg_preds"]
        centernesses = pred_dict["center_preds"]
        featmap_size = cls_scores.size()[-2:]

        points = self.get_points(featmap_size, bbox_preds.dtype,
                                      bbox_preds.device)
        mlvl_points=[points]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[img_id].detach()
            ]
            bbox_pred_list = [
                bbox_preds[img_id].detach()
            ]
            centerness_pred_list = [
                centernesses[img_id].detach()
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def loss(self,
             pred_dict,
             gt_labels,
             gt_bboxes,
             img_metas,
             pad_shape=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """

        # assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        cls_scores=pred_dict["cls_preds"]
        featmap_sizes = cls_scores.size()[-2:]
        points = self.get_points(featmap_sizes,self.stride, cls_scores.dtype,
                                           cls_scores.device)
        labels, bbox_targets = self.get_targets(points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores.size(0)
        flatten_cls_scores=cls_scores.permute(0, 2, 3, 1).reshape(-1, self.num_class)
        flatten_labels = labels
        # flatten_bbox_targets = bbox_targets

        # labels_test=flatten_labels.reshape(featmap_sizes)
        # print("label test shape is ",labels_test.shape)
        # neg_inds = (flatten_labels >= self.num_class).nonzero().reshape(-1)
        # centerness_targets_test = self.centerness_target(bbox_targets)
        # centerness_targets_test[neg_inds]=0
        # centerness_targets_test=centerness_targets_test.reshape(featmap_sizes)
        #
        # # centerness_targets_test[]
        # plt.figure("Image")  # 图像窗口名称
        # plt.get_current_fig_manager().window.showMaximized()
        #
        #
        # plt.subplot(1,2,1)
        # plt.imshow((labels_test).cpu().numpy(), cmap='jet')
        # plt.axis('on')  # 关掉坐标轴为 off
        # plt.title('label')  # 图像题目
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow((centerness_targets_test).cpu().numpy(), cmap='jet')
        # plt.axis('on')  # 关掉坐标轴为 off
        # plt.title('center')  # 图像题目
        #
        # plt.show()


        # repeat points to align with bbox_preds
        # flatten_points = points.repeat(num_imgs, 1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_class
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        return dict(
            loss_cls_2d=loss_cls,
            # loss_bbox_2d=loss_bbox,
            # loss_centerness_2d=loss_centerness
        )

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (Tensor): Points, shape (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (Tensor): Labels . \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        # assert len(points) == len(self.regress_ranges)
        # num_levels = len(points)
        # expand regress ranges to align with points
        # expanded_regress_ranges = [
        #     points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
        #         points[i]) for i in range(num_levels)
        # ]
        # concat all levels points and regress ranges
        # concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        # concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        # num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=points,)


        # split to per img, per level
        # labels_list = [labels.split(num_points, 0) for labels in labels_list]
        # bbox_targets_list = [
        #     bbox_targets.split(num_points, 0)
        #     for bbox_targets in bbox_targets_list
        # ]
        concat_labels=torch.cat(labels_list,dim=0)
        concat_bbox_targets=torch.cat(bbox_targets_list,dim=0)
        # concat per level image
        # concat_lvl_labels = []
        # concat_lvl_bbox_targets = []
        # for i in range(num_levels):
        #     concat_lvl_labels.append(
        #         torch.cat([labels[i] for labels in labels_list]))
        #     bbox_targets = torch.cat(
        #         [bbox_targets[i] for bbox_targets in bbox_targets_list])
        #     if self.norm_on_bbox:
        #         bbox_targets = bbox_targets / self.strides[i]
        #     concat_lvl_bbox_targets.append(bbox_targets)
        return concat_labels,concat_bbox_targets


    def _get_target_single(self, gt_bboxes, gt_labels, points):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_class), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)


        # condition1: inside a `center bbox`
        if self.center_sampling:
            radius = self.center_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_ones(center_xs.shape) * self.stride * radius

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

            # condition2: limit the regression range for each location
            # max_regress_distance = bbox_targets.max(-1)[0]
            # inside_regress_range = (
            #         (max_regress_distance >= regress_ranges[..., 0])
            #         & (max_regress_distance <= regress_ranges[..., 1]))

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        areas[inside_gt_bbox_mask == 0] = INF
        # areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_class  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                                     top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


    def get_points(self,featmap_size,stride,dtype,device):
        """Get points according to feature map sizes."""
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points