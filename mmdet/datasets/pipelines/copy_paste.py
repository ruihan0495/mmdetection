import numpy as np
import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian
from ..builder import PIPELINES
from transfroms import Albu


import mmcv
import numpy as np
import os
import cv2
import random
import numpy as np
import albumentations as A
from copy import deepcopy
from skimage.filters import gaussian
from ..builder import PIPELINES


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img

def mask_copy_paste(mask, paste_mask, alpha):
    raise NotImplementedError

def masks_copy_paste(masks, paste_masks, alpha):
    if alpha is not None:
        #eliminate pixels that will be pasted over
        masks = [
            np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.uint8) for mask in masks
        ]
        masks.extend(paste_masks)

    return masks

def extract_bboxes(masks):
    bboxes = []
    h, w = masks[0].shape
    for mask in masks:
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1
            y1 /= w
            y2 /= w
            x1 /= h
            x2 /= h
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0

        bboxes.append((y1, x1, y2, x2))

    return bboxes

def bboxes_copy_paste(bboxes, paste_bboxes, masks, paste_masks, alpha, key):
    if key == 'paste_bboxes':
        return bboxes
    elif paste_bboxes is not None:
        masks = masks_copy_paste(masks, paste_masks=[], alpha=alpha)
        adjusted_bboxes = extract_bboxes(masks)

        #only keep the bounding boxes for objects listed in bboxes
        mask_indices = [box[-1] for box in bboxes]
        adjusted_bboxes = [adjusted_bboxes[idx] for idx in mask_indices]
        #append bbox tails (classes, etc.)
        adjusted_bboxes = [bbox + tail[4:] for bbox, tail in zip(adjusted_bboxes, bboxes)]

        #adjust paste_bboxes mask indices to avoid overlap
        if len(masks) > 0:
            max_mask_index = len(masks)
        else:
            max_mask_index = 0

        paste_mask_indices = [max_mask_index + ix for ix in range(len(paste_bboxes))]
        paste_bboxes = [pbox[:-1] + (pmi,) for pbox, pmi in zip(paste_bboxes, paste_mask_indices)]
        adjusted_paste_bboxes = extract_bboxes(paste_masks)
        adjusted_paste_bboxes = [apbox + tail[4:] for apbox, tail in zip(adjusted_paste_bboxes, paste_bboxes)]

        bboxes = adjusted_bboxes + adjusted_paste_bboxes

    return bboxes

def keypoints_copy_paste(keypoints, paste_keypoints, alpha):
    #remove occluded keypoints
    if alpha is not None:
        visible_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            tail = kp[2:]
            if alpha[int(y), int(x)] == 0:
                visible_keypoints.append(kp)

        keypoints = visible_keypoints + paste_keypoints

    return keypoints

class CopyPaste(A.DualTransform):
    def __init__(
        self,
        blend=True,
        sigma=3,
        pct_objects_paste=0.1,
        max_paste_objects=None,
        bbox_params=None,
        p=0.5,
        always_apply=False
    ):
        super(CopyPaste, self).__init__(always_apply, p)
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects
        self.p = p
        self.always_apply = always_apply
        self.bbox_params = bbox_params

    @staticmethod
    def get_class_fullname():
        return 'copypaste.CopyPaste'

    @property
    def targets_as_params(self):
        return [
            "masks",
            "paste_image",
            #"paste_mask",
            "paste_masks",
            "paste_bboxes",
            #"paste_keypoints"
        ]

    def get_params_dependent_on_targets(self, params):
        image = params["paste_image"]
        masks = None
        if "paste_mask" in params:
            #handle a single segmentation mask with
            #multiple targets
            #nothing for now.
            raise NotImplementedError
        elif "paste_masks" in params:
            masks = params["paste_masks"]

        assert(masks is not None), "Masks cannot be None!"

        bboxes = params.get("paste_bboxes", None)
        keypoints = params.get("paste_keypoints", None)

        #number of objects: n_bboxes <= n_masks because of automatic removal
        n_objects = len(bboxes) if bboxes is not None else len(masks)

        #paste all objects if no restrictions
        n_select = n_objects
        if self.pct_objects_paste:
            n_select = int(n_select * self.pct_objects_paste)
        if self.max_paste_objects:
            n_select = min(n_select, self.max_paste_objects)

        #no objects condition
        if n_select == 0:
            return {
                "param_masks": params["masks"],
                "paste_img": None,
                "alpha": None,
                "paste_mask": None,
                "paste_masks": None,
                "paste_bboxes": None,
                "paste_keypoints": None,
                "objs_to_paste": []
            }

        #select objects
        objs_to_paste = np.random.choice(
            range(0, n_objects), size=n_select, replace=False
        )

        #take the bboxes
        if bboxes:
            bboxes = [bboxes[idx] for idx in objs_to_paste]
            #the last label in bboxes is the index of corresponding mask
            mask_indices = [bbox[-1] for bbox in bboxes]

        #create alpha by combining all the objects into
        #a single binary mask
        masks = [masks[idx] for idx in mask_indices]

        alpha = masks[0] > 0
        for mask in masks[1:]:
            alpha += mask > 0

        return {
            "param_masks": params["masks"],
            "paste_img": image,
            "alpha": alpha,
            "paste_mask": None,
            "paste_masks": masks,
            "paste_bboxes": bboxes,
            "paste_keypoints": keypoints
        }

    @property
    def ignore_kwargs(self):
        return [
            "paste_image",
            "paste_mask",
            "paste_masks"
        ]

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None and key not in self.ignore_kwargs:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                target_dependencies['key'] = key
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def apply(self, img, paste_img, alpha, **params):
        return image_copy_paste(
            img, paste_img, alpha, blend=self.blend, sigma=self.sigma
        )

    def apply_to_mask(self, mask, paste_mask, alpha, **params):
        return mask_copy_paste(mask, paste_mask, alpha)

    def apply_to_masks(self, masks, paste_masks, alpha, **params):
        return masks_copy_paste(masks, paste_masks, alpha)

    def apply_to_bboxes(self, bboxes, paste_bboxes, param_masks, paste_masks, alpha, key, **params):
        return bboxes_copy_paste(bboxes, paste_bboxes, param_masks, paste_masks, alpha, key)

    def apply_to_keypoints(self, keypoints, paste_keypoints, alpha, **params):
        raise NotImplementedError
        #return keypoints_copy_paste(keypoints, paste_keypoints, alpha)

    def get_transform_init_args_names(self):
        return (
            "blend",
            "sigma",
            "pct_objects_paste",
            "max_paste_objects"
        )  

@PIPELINES.register_module()
class AlbuCopyPaste(Albu):
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self,
                 bbox_params=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')
        transforms = [dict(type='CopyPaste',
                            blend=True,
                            sigma=1,
                            pct_objects_paste=0.5,
                            p=1)]
        keymap = {'img': 'paste_image',
                  'gt_masks': 'paste_masks',
                  'gt_bboxes': 'paste_bboxes'}
        super(AlbuCopyPaste, self).__init__(transforms, keymap)

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose(self.albu_builder(self.transforms[0]),
                           bbox_params=self.bbox_params)

        self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type) and obj_type == "CopyPaste":
            obj_cls = obj_type
        else:
            raise TypeError(
                f'Invalid CopyPaste transformation')
        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)




