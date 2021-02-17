_base_ = '../resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py'
model = dict(
    roi_head = dict(bbox_head=dict(num_classes=13),
                    mask_head=dict(num_classes=13)))

dataset_type = 'DeepFashion2Dataset'
data_root = 'data/DeepFashion2/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='sample_train/deepfashion2.json',
        img_prefix='sample_train/image/',
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file='sample_train/deepfashion2.json',
        img_prefix='sample_train/image/',
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file='deepfashion2.json',
        img_prefix='image/',
        data_root=data_root))
total_epochs = 2
evaluation = dict(interval=5, metric=['bbox', 'segm'])
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201122_104428-99eca4c7.pth'