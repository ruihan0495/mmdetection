_base_ = '../hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco.py'

model = dict(
    roi_head = dict(
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
        mask_head = dict(num_classes=13)
    )
)

dataset_type = 'DeepFashion2Dataset'
data_root = 'data/DeepFashion2/'
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='sample_train/deepfashion2.json',
        img_prefix='sample_train/image/',
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file='sample_val/deepfashion2.json',
        img_prefix='sample_val/image/',
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file='sample_val/deepfashion2.json',
        img_prefix='sample_val/image/',
        data_root=data_root))
evaluation = dict(interval=5, metric=['bbox', 'segm']) 
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w40_20e_coco/htc_hrnetv2p_w40_20e_coco_20200529_183411-417c4d5b.pth'