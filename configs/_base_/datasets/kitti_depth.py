# dataset settings
dataset_type = 'KittiDepthDataset'
data_root = 'data/kitti_raw/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (352, 1216)
input_size = (176, 608)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile'),
    dict(type='Crop',crop_size=crop_size),
    dict(type='Resize', img_scale=input_size, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=input_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_depth','ref_img','pose','ref_pose','cam_K','imu2cam','ref_seq_id']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=input_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img','gt_depth']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ref_seq_id=[-1, 1],
        img_suffix='.png',
        img_idx_file='data_splits/custom_all_imgs.txt',
        idx_file='data_splits/custom_train_index.txt',
        img_dir='image_02/data',
        depth_dir='proj_depth/velodyne/image_02',
        depth_suffix='.npz',
        pose_file='image_02/poses.txt',
        cam_intrinc_file='image_02/cam.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_idx_file='data_splits/custom_all_imgs.txt',
        idx_file='data_splits/custom_val_index.txt',
        img_suffix='.png',
        img_dir='image_02/data',
        depth_suffix='.npz',
        depth_dir='proj_depth/velodyne/image_02',
        pipeline=test_pipeline),
    test=dict(
        test_mode=True,
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        img_dir='image_02/data',
        img_idx_file='data_splits/custom_all_imgs.txt',
        idx_file='data_splits/custom_test_index.txt',
        depth_dir='proj_depth/groundtruth/image_02',
        pipeline=test_pipeline))
