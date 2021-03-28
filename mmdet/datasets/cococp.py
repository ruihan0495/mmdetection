import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict
import random

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .coco import CocoDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')

@DATASETS.register_module()
class CocoDatasetCP(CocoDataset):
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h, ann['category_id'], i]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 6), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 6), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            bboxes_ignore=gt_bboxes_ignore,
            labels=gt_labels,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            img_info = self.data_infos[idx]
            ann_info = self.get_ann_info(idx)
            data = dict(img_info=img_info, ann_info=ann_info)
            if data is None:
                idx = self._rand_another(idx)
                continue
            paste_idx = random.randint(0, self.__len__()-1)
            img_info = self.data_infos[paste_idx]
            ann_info = self.get_ann_info(paste_idx)
            paste_img_data = dict(img_info=img_info, ann_info=ann_info)

            data['paste_masks'] = paste_img_data['ann_info']['masks']
            data['paste_bboxes'] = paste_img_data['ann_info']['bboxes']
            data['paste_image_info'] = paste_img_data['img_info']
            labels_orig = data['ann_info']['labels']
            labels_paste = paste_img_data['ann_info']['labels']
            data['ann_info']['labels'] = np.concatenate((labels_orig, labels_paste), axis=None)
            self.pre_pipeline(data)
            return self.pipeline(data)