import torch.nn as nn
import effdet

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

checkpoint_path = None
num_classes = 13
pretrained_backbone = True
pretrained = True
redundant_bias = None
smoothing = None
legacy_focal = None
jit_loss = None
soft_nms = True
bench_labeler = None
initial_checkpoint = None


@DETECTORS.register_module()
class SingleStageInsDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageInsDetector, self).__init__()
        self.eff_backbone_flag = False if 'efficient' not in backbone['type'] else True

        if self.eff_backbone_flag == False:
            self.backbone = build_backbone(backbone)
        else:
            '''
            self.backbone = effdet.create_model(args.model,
                                                bench_task='train',
                                                num_classes=args.num_classes,
                                                pretrained=args.pretrained,
                                                pretrained_backbone=args.pretrained_backbone,
                                                redundant_bias=args.redundant_bias,
                                                label_smoothing=args.smoothing,
                                                legacy_focal=args.legacy_focal,
                                                jit_loss=args.jit_loss,
                                                soft_nms=args.soft_nms,
                                                bench_labeler=args.bench_labeler,
                                                checkpoint_path=args.initial_checkpoint,
                                                )
            '''
            self.backbone = effdet.create_model('tf_efficientdet_d1',
                                                bench_task='train',
                                                num_classes=num_classes,
                                                pretrained=pretrained,
                                                pretrained_backbone=pretrained_backbone,
                                                redundant_bias=redundant_bias,
                                                label_smoothing=smoothing,
                                                legacy_focal=legacy_focal,
                                                jit_loss=jit_loss,
                                                soft_nms=soft_nms,
                                                bench_labeler=bench_labeler,
                                                checkpoint_path=initial_checkpoint,
                                                )
        self.with_neck = False
        self.with_neck = False
        
        if neck is not None:
            self.neck = build_neck(neck)
            self.with_neck = True
        if mask_feat_head is not None:
            self.mask_feat_head = build_head(mask_feat_head)
            self.with_mask_feat_head = True
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageInsDetector, self).init_weights(pretrained)
        if not self.eff_backbone_flag: 
            self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_mask_feat_head:
            if isinstance(self.mask_feat_head, nn.Sequential):
                for m in self.mask_feat_head:
                    m.init_weights()
            else:
                self.mask_feat_head.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
            loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, eval=True)

        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
            seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)
        else:
            seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result  

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError