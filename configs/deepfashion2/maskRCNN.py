  
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'
model = dict(pretrained='torchvision://resnet101', 
            backbone=dict(depth=101),
            roi_head=dict(
                type='StandardRoIHead',
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                bbox_head=dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=13,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=False,
                    loss_cls=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                mask_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                mask_head=dict(
                    type='FCNMaskHead',
                    num_convs=4,
                    in_channels=256,
                    conv_out_channels=13,
                    num_classes=13,
                    loss_mask=dict(
                        type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
total_epochs = 2

# dataset settings
dataset_type = 'DeepFashion2Dataset'
data_root = 'data/DeepFashion2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(750, 1101), keep_ratio=True),
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
        img_scale=(750, 1101),
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
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='deepfashion2.json',
        img_prefix='image/',
        pipeline=train_pipeline,
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file='deepfashion2.json',
        img_prefix='image/',
        pipeline=test_pipeline,
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file='deepfashion2.json',
        img_prefix='image/',
        pipeline=test_pipeline,
        data_root=data_root))
evaluation = dict(interval=5, metric=['bbox', 'segm'])
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth'