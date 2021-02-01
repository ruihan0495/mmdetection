_base_ = ['../detectors/htc_r50_rfp_1x_coco.py',
          '../_base_/datasets/deepfashion2.py']

model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=13,
            ignore_label=255,
            loss_weight=0.2)))
# runtime settings
total_epochs = 15