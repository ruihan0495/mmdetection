import argparse
import os
import json
import numpy as np
from mmdet.datasets import build_dataset
from mmdet.apis import inference_detector, init_detector

from mmcv import Config
from mmdet.core.evaluation import eval_map


def json_to_annotation(json_name):    
    annotation = {}
    bboxes = []
    labels = []
    with open(json_name, 'r') as f:
        temp = json.loads(f.read())
        for i in temp:
            if i == 'source' or i=='pair_id':
                continue
            bboxes.append(temp['bounding_box'])
            labels.append(temp['category_id'])
        annotation['bboxes'] = np.asarray(bboxes)
        annotation['labels'] = np.asarray(labels)
        annotation['image_id'] = json_name.split('.')[0]
    return [annotation]


def ap_for_image(cfg, checkpoint, image_dir, anno_dir, out_dir):
    model = init_detector(cfg, checkpoint, device='cuda:0')
    images = os.listdir(image_dir)
    anno_files = os.listdir(anno_dir)
    output_name = 'precision.json'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_name = os.path.join(out_dir, output_name)
    for image, anno in zip(images, anno_files):
        assert image.split('.')[0] == anno.split('.')[0], "image and annotation mismatch!"
        image_path = os.path.join(image_dir, image)
        json_name = os.path.join(anno_dir, anno)
        results = inference_detector(model, image_path)
        annotations = json_to_annotation(json_name)
        mean_ap, _ = eval_map(results[0], annotations)
        with open(output_name, 'w') as f:
            json.dump([annotations['image_id'], mean_ap], f)


if __name__ == "__main__":
    image_dir = ""
    anno_dir = ""
    out_dir = ""
    config = ""
    checkpoint = ""

    ap_for_image(config, checkpoint, image_dir, anno_dir, out_dir)

    


