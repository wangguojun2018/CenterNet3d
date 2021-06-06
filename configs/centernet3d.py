# dataset settings
_base_ = [ './_base_/schedules/cyclic_40e.py',
    './_base_/default_runtime.py']
dataset_type = 'KittiDataset'
data_root = '/home/wangguojun/dataset/kitti/object/'
class_names = ['Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size=[0.05, 0.05, 0.1]
num_class=1
checkpoint_config = dict(interval=2)
evaluation = dict(interval=5)
lr = 0.000225
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
total_epochs = 50
input_modality = dict(use_lidar=True, use_camera=False)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15))

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_test.pkl',
        split='testing',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

model = dict(
    type='CenterNet3D',
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoderV2',
        in_channels=4,
        sparse_shape=[40, 1600, 1408],
        out_channels=320),
    backbone=dict(
        type='SECONDFPNDCN',
        in_channels=128,
        layer_nums=[3],
        layer_strides=[1],
        num_filters=[128],
        upsample_strides=[2],
        out_channels=[128],
    ),

    bbox_head=dict(
        type='Center3DHead',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        bbox_coder=dict(type='Center3DBoxCoder',num_class=num_class,
                        voxel_size=voxel_size,pc_range=point_cloud_range,
                        num_dir_bins=0,
                        downsample_ratio=4.0,
                        min_overlap=0.01,
                        keypoint_sensitive=True,
                        ),
        loss_cls=dict(type='MSELoss',loss_weight=1.0),
        loss_xy=dict(type='GatherBalancedL1Loss',loss_weight=1.0),
        loss_z=dict(type='GatherBalancedL1Loss', loss_weight=1.0),
        loss_dim=dict(type='GatherBalancedL1Loss', loss_weight=2.0),
        loss_dir=dict(type='GatherBalancedL1Loss', loss_weight=0.5),
        # loss_decode=dict(type='Boxes3dDecodeLoss', loss_weight=0.5),
        bias_cls=-7.94,
        loss_corner=dict(type='MSELoss', loss_weight=1.0),
        )
)
# model training and testing settings
train_cfg = dict()
test_cfg = dict(
    score_thr=0.10,
)
find_unused_parameters=True

