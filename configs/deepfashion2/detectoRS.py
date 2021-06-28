_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict( 
      backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
      neck=dict(
          type='RFP',
          rfp_steps=2,
          aspp_out_channels=64,
          aspp_dilations=(1, 3, 6, 1),
          rfp_backbone=dict(
              rfp_inplanes=256,
              type='DetectoRS_ResNet',
              depth=50,
              num_stages=4,
              out_indices=(0, 1, 2, 3),
              frozen_stages=1,
              norm_cfg=dict(type='BN', requires_grad=True),
              norm_eval=True,
              conv_cfg=dict(type='ConvAWS'),
              sac=dict(type='SAC', use_deform=True),
              stage_with_sac=(False, True, True, True),
              pretrained='torchvision://resnet50',
              style='pytorch')),
      roi_head=dict(
          type='CascadeRoIHead',
          num_stages=3,
          stage_loss_weights=[1, 0.5, 0.25],
          bbox_roi_extractor=dict(
              type='SingleRoIExtractor',
              roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
              out_channels=256,
              featmap_strides=[4, 8, 16, 32]),
          bbox_head=[
              dict(
                  type='Shared2FCBBoxHead',
                  in_channels=256,
                  fc_out_channels=1024,
                  roi_feat_size=7,
                  num_classes=13,
                  bbox_coder=dict(
                      type='DeltaXYWHBBoxCoder',
                      target_means=[0., 0., 0., 0.],
                      target_stds=[0.1, 0.1, 0.2, 0.2]),
                  reg_class_agnostic=True,
                  loss_cls=dict(
                      type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_weight=1.0),
                  loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                 loss_weight=1.0)),
              dict(
                  type='Shared2FCBBoxHead',
                  in_channels=256,
                  fc_out_channels=1024,
                  roi_feat_size=7,
                  num_classes=13,
                  bbox_coder=dict(
                      type='DeltaXYWHBBoxCoder',
                      target_means=[0., 0., 0., 0.],
                      target_stds=[0.05, 0.05, 0.1, 0.1]),
                  reg_class_agnostic=True,
                  loss_cls=dict(
                      type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_weight=1.0),
                  loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                 loss_weight=1.0)),
              dict(
                  type='Shared2FCBBoxHead',
                  in_channels=256,
                  fc_out_channels=1024,
                  roi_feat_size=7,
                  num_classes=13,
                  bbox_coder=dict(
                      type='DeltaXYWHBBoxCoder',
                      target_means=[0., 0., 0., 0.],
                      target_stds=[0.033, 0.033, 0.067, 0.067]),
                  reg_class_agnostic=True,
                  loss_cls=dict(
                      type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_weight=1.0),
                  loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
          ],
          mask_roi_extractor=dict(
              type='SingleRoIExtractor',
              roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
              out_channels=256,
              featmap_strides=[4, 8, 16, 32]),
          mask_head=dict(
              type='FCNMaskHead',
              num_convs=4,
              in_channels=256,
              conv_out_channels=256,
              num_classes=13,
              loss_mask=dict(
                  type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
# runtime settings
total_epochs = 20

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35.0, norm_type=2))
# dataset settings
dataset_type = 'DeepFashion2Dataset'
data_root = 'data/DeepFashion2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='sample_train/deepfashion2.json',
        img_prefix='sample_train/image/',
        pipeline=train_pipeline,
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file='sample_val/deepfashion2.json',
        img_prefix='sample_val/image/',
        pipeline=test_pipeline,
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file='sample_val/deepfashion2.json',
        img_prefix='sample_val/image/',
        pipeline=test_pipeline,
        data_root=data_root))
evaluation = dict(interval=1, metric=['bbox', 'segm'])
checkpoint_config = dict(interval=4)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
work_dir = './work_dirs/detectoRS'
resume_from = None
load_from = None