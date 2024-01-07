# dataset settings
dataset_type = 'NyuDataset_al'
data_root = '/data/xuj/workspace/dataset/NYUv2/mmseg_version'
kp_infor_dir = '/data/xuj/workspace/dataset/NYUv2/mmseg_version/images/train/patch_ids.json'
img_norm_cfg = dict(
    mean=[123.55185386, 106.10660834, 101.43416643], std=[73.27660521, 75.08845355, 78.38360167], to_rgb=True)
crop_size = (384, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='Resize', img_scale=(480, 640), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='Resize', img_scale=(480, 640)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        kp_infor_dir=kp_infor_dir,
        img_dir='images/train',
        ann_dir='labels/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=test_pipeline))
