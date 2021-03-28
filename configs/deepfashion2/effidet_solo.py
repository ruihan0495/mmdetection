model = dict(
    type='SingleStageInsDetector',
    backbone=dict(
        type='tf_efficientdet_d1',
        style='pytorch'),
    bbox_head=dict(
        type='SOLOv2Head',
        num_classes=81,
        in_channels=256,
        stacked_convs=2,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=128,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    mask_feat_head=dict(
            type='MaskFeatHead',
            in_channels=256,
            out_channels=128,
            start_level=0,
            end_level=3,
            num_classes=128,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    )

train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)

dataset_type = 'DeepFashion2Dataset'
data_root = '/content/gdrive/MyDrive/sample_val'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(768, 512), (768, 480), (768, 448),
                   (768, 416), (768, 384), (768, 352)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(768, 448),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='deepfashion2.json',
        img_prefix='image/',
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file='deepfashion2.json',
        img_prefix='image/',
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file='deepfashion2.json',
        img_prefix='image/',
        data_root=data_root))

total_epochs = 36

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[27, 33])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/solov2_light_release_r50_fpn_8gpu_3x'
load_from = None

evaluation = dict(interval=5, metric=['bbox', 'segm'])
resume_from = None
workflow = [('train', 1)]
