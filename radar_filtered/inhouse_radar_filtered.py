voxel_size = [0.16, 0.16, 4]
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2],
        voxel_size=[0.16, 0.16, 4],
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=6,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2]),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -39.68, -0.6, 70.4, 39.68, -0.6],
                    [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                    [0, -39.68, -1.78, 70.4, 39.68, -1.78]],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        assigner=[
            dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
_point_cloud_range = [0, -39.68, -2, 69.12, 39.68, 2]
dataset_type = 'InhouseDataset'
data_root = 'data/inhouse_filtered_radar/kitti_format/'
pointcloud_dir = 'radar'
load_dim = 6
use_dim = 6
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -39.68, -2, 69.12, 39.68, 2]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root='data/inhouse_filtered_radar/kitti_format/',
    info_path=
    'data/inhouse_filtered_radar/kitti_format/inhouse_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=['Pedestrian', 'Cyclist', 'Car'],
    sample_groups=dict(Car=15, Pedestrian=0, Cyclist=0),
    points_loader=dict(
        type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=6))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=6),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='data/inhouse_filtered_radar/kitti_format/',
            info_path=
            'data/inhouse_filtered_radar/kitti_format/inhouse_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
            classes=['Pedestrian', 'Cyclist', 'Car'],
            sample_groups=dict(Car=15, Pedestrian=0, Cyclist=0),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=6,
                use_dim=6))),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2]),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=6),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=6,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car'],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
train_dataset = dict(
    type='InhouseDataset',
    data_root='data/inhouse_filtered_radar/kitti_format/',
    ann_file='data/inhouse_filtered_radar/kitti_format/inhouse_infos_train.pkl',
    split='training',
    pts_prefix='radar',
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=6,
            use_dim=6,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            file_client_args=dict(backend='disk')),
        dict(
            type='ObjectSample',
            db_sampler=dict(
                data_root='data/inhouse_filtered_radar/kitti_format/',
                info_path=
                'data/inhouse_filtered_radar/kitti_format/inhouse_dbinfos_train.pkl',
                rate=1.0,
                prepare=dict(
                    filter_by_difficulty=[-1],
                    filter_by_min_points=dict(
                        Car=5, Pedestrian=10, Cyclist=10)),
                classes=['Pedestrian', 'Cyclist', 'Car'],
                sample_groups=dict(Car=12, Pedestrian=0, Cyclist=0),
                points_loader=dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=6))),
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
        dict(
            type='PointsRangeFilter',
            point_cloud_range=[0, -40, -2, 70.4, 40, 2]),
        dict(
            type='ObjectRangeFilter',
            point_cloud_range=[0, -40, -2, 70.4, 40, 2]),
        dict(type='PointShuffle'),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Pedestrian', 'Cyclist', 'Car']),
        dict(
            type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ],
    modality=dict(use_lidar=True, use_camera=False),
    classes=['Pedestrian', 'Cyclist', 'Car'],
    test_mode=False,
    box_type_3d='LiDAR')
data = dict(
    samples_per_gpu=14,
    workers_per_gpu=4,
    train=dict(
        type='InhouseDataset',
        data_root='data/inhouse_filtered_radar/kitti_format/',
        ann_file=
        'data/inhouse_filtered_radar/kitti_format/inhouse_infos_train.pkl',
        split='training',
        pts_prefix='radar',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=6,
                use_dim=6),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='ObjectSample',
                db_sampler=dict(
                    data_root='data/inhouse_filtered_radar/kitti_format/',
                    info_path=
                    'data/inhouse_filtered_radar/kitti_format/inhouse_dbinfos_train.pkl',
                    rate=1.0,
                    prepare=dict(
                        filter_by_difficulty=[-1],
                        filter_by_min_points=dict(
                            Car=5, Pedestrian=10, Cyclist=10)),
                    classes=['Pedestrian', 'Cyclist', 'Car'],
                    sample_groups=dict(Car=15, Pedestrian=0, Cyclist=0),
                    points_loader=dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=6,
                        use_dim=6))),
            dict(
                type='ObjectNoise',
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                global_rot_range=[0.0, 0.0],
                rot_range=[-0.15707963267, 0.15707963267]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05]),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car']),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type='InhouseDataset',
        data_root='data/inhouse_filtered_radar/kitti_format/',
        ann_file=
        'data/inhouse_filtered_radar/kitti_format/inhouse_infos_val.pkl',
        split='training',
        pts_prefix='radar',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=6,
                use_dim=6),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian', 'Cyclist', 'Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='InhouseDataset',
        data_root='data/inhouse_filtered_radar/kitti_format/',
        ann_file=
        'data/inhouse_filtered_radar/kitti_format/inhouse_infos_val.pkl',
        split='training',
        pts_prefix='radar',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=6,
                use_dim=6),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -39.68, -2, 69.12, 39.68, 2]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian', 'Cyclist', 'Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=4,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=6,
            use_dim=6,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Pedestrian', 'Cyclist', 'Car'],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
lr = 0.001
optimizer = dict(type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=160)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'radar_filtered'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = [0]
