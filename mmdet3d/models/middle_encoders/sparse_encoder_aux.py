import spconv
from torch import nn
import torch
from mmdet3d.models.registry import MIDDLE_ENCODERS
from mmdet3d.ops import three_interpolate, three_nn
from torchplus.tools import change_default_args
from ..losses.center_loss import weighted_sigmoid_focal_loss,weighted_smoothl1

BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)


@MIDDLE_ENCODERS.register_module()
class SparseEncoder_AUX(nn.Module):
    def __init__(self,in_channels=4,sparse_shape=[40, 1600, 1408],out_channels=64):

        super(SparseEncoder_AUX, self).__init__()


        self.sparse_shape = sparse_shape
        print(self.sparse_shape)
        self.backbone = VxNet(in_channels,out_channels)
        self.num_input_features=in_channels

    def forward(self, voxel_features, coors, batch_size):

        points_mean=voxel_features.new_zeros((voxel_features.shape[0],4))
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]


        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x, point_misc = self.backbone(x, points_mean)

        x = x.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)

        return x, point_misc


    def build_aux_target(self, points, gt_boxes3d):
        center_offsets = []
        pts_labels = []
        for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i]
            boxes3d = boxes3d.to(points.device)
            idx = torch.nonzero(points[:, 0] == i)[:,0]
            xyz = points[idx, 1:]
            # print("xyz shape",xyz.shape)
            cls_label = xyz.new_zeros(xyz.shape[0])
            reg_label = xyz.new_zeros((xyz.shape[0], 3))
            points_box_id = boxes3d.points_in_boxes(xyz).long()

            for j in range(boxes3d.tensor.shape[0]):
                fg_pts_rect = xyz[points_box_id == j]
                cls_label[points_box_id == j] = 1
                center3d = boxes3d.gravity_center[j:j+1]  # (x, y, z)
                # idx_box=torch.nonzero(points_box_id==j)
                reg_label[points_box_id == j] = center3d - fg_pts_rect

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(cls_label)
            center_offsets.append(reg_label)

        center_offsets = torch.cat(center_offsets)
        pts_labels = torch.cat(pts_labels)

        return pts_labels, center_offsets

    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):

        N = len(gt_bboxes)

        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return dict(
            aux_loss_cls = aux_loss_cls,
            aux_loss_reg = aux_loss_reg,
        )

class VxNet(nn.Module):

    def __init__(self, num_input_features,num_out_features=64):
        super(VxNet, self).__init__()
        #[40,1600,1408]


        self.activation_fcn=change_default_args(negative_slope=0.01,inplace=True)(nn.LeakyReLU)

        self.extra_conv = spconv.SparseSequential(
            SpConv3d(64, 64, (3,1,1), (2,1,1), indice_key="down3"),
            BatchNorm1d(64),
            self.activation_fcn())


        self.conv0 = double_conv(num_input_features, 16, 'subm0',activation=self.activation_fcn)
        self.down0 = stride_conv(16, 32, 'down0',activation=self.activation_fcn)

        self.conv1 = double_conv(32, 32, 'subm1',activation=self.activation_fcn) #[20,800,704]
        self.down1 = stride_conv(32, 64, 'down1',activation=self.activation_fcn)

        self.conv2 = triple_conv(64, 64, 'subm2',activation=self.activation_fcn) #[10,400,352]
        self.down2 = stride_conv(64, 64, 'down2',activation=self.activation_fcn)

        self.conv3 = triple_conv(64, 64, 'subm3',activation=self.activation_fcn)  # #[5,200,176]


        self.point_fc = nn.Linear(160, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)


    def forward(self, x, points_mean):

        x = self.conv0(x)
        x = self.down0(x)  # sp
        x = self.conv1(x)  # 2x sub

        if self.training:
            # 根据体素的gridmap坐标计算体素在点云空间中对应的点坐标
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.1, .1, .2))
            # 根据降采样后每个体素中心的xyz坐标计算得到全部体素xyz均值处的特征
            p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)


        x = self.down1(x)
        x = self.conv2(x)

        if self.training:
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.2, .2, .4))
            p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

        x = self.down2(x)
        x = self.conv3(x)

        if self.training:
            vx_feat, vx_nxyz = tensor2points(x, voxel_size=(.4, .4, .8))
            p3 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

        out = self.extra_conv(x)

        if not self.training:
            return out, None

        pointwise = self.point_fc(torch.cat([p1, p2, p3], dim=-1))
        point_cls = self.point_cls(pointwise)
        point_reg = self.point_reg(pointwise)
        return out, (points_mean, point_cls, point_reg)




def single_conv(in_channels, out_channels, indice_key=None,activation=None):
    return spconv.SparseSequential(
        SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key),
        BatchNorm1d(out_channels),
       activation())


def double_conv(in_channels, out_channels, indice_key=None,activation=None):
    return spconv.SparseSequential(
            SubMConv3d(in_channels, out_channels,3,indice_key=indice_key),
            BatchNorm1d(out_channels),
            activation(),
            SubMConv3d(out_channels, out_channels, 3,indice_key=indice_key),
            BatchNorm1d(out_channels),
            activation())



def triple_conv(in_channels, out_channels, indice_key=None,activation=None):
    return spconv.SparseSequential(
            SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key),
            BatchNorm1d(out_channels),
            activation(),
            SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key),
            BatchNorm1d(out_channels),
            activation(),
            SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key),
            BatchNorm1d(out_channels,),
            activation(),
    )

def stride_conv(in_channels, out_channels, indice_key=None,activation=None):

    return spconv.SparseSequential(
            SpConv3d(in_channels, out_channels, 3, 2, padding=1, indice_key=indice_key),
            BatchNorm1d(out_channels),
            activation())

def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param unknown: (n, 4) 每个体素的xyz均值
    :param known: (m, 4) 体素中心对应的xyz坐标
    :param known_feats: (m, C) 体素的特征
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    unknown=unknown.unsqueeze(0)
    known=known.unsqueeze(0)
    known_feats=known_feats.unsqueeze(0).permute(0,2,1).contiguous()
    dist, idx = three_nn(unknown, known)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = three_interpolate(known_feats, idx, weight)
    interpolated_feats=interpolated_feats.permute(0,2,1).contiguous().squeeze(0)
    return interpolated_feats


def tensor2points(tensor, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
    #根据体素的坐标计算体素在点云空间中对应的点坐标
    indices = tensor.indices.float()  #coordinate
    offset = torch.Tensor(offset).to(indices.device)
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size
    return tensor.features, indices