_base_ = '../detr/detr_r50_8x2_150e_coco.py'

model = dict(
    bbox_head = dict(num_classes=13))

dataset_type = 'DeepFashion2Dataset'
data_root = 'data/DeepFashion2/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
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
total_epochs = 2
evaluation = dict(interval=5, metric=['bbox', 'segm'])
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'