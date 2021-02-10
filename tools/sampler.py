'''
Randonly sample 10% of the training, val, and test data 
from DeepFashion2 dataset.
'''
import os
import numpy as np
import shutil
import math

def sampler(image_dir, anno_dir, ratio, save_img_dir, save_anno_dir):
    image_files = os.listdir(image_dir)
    anno_files = os.listdir(anno_dir)
    total_files = len(image_files)
    assert len(image_files) == len(anno_files), "Annotations and images don't match!"
    sample_num_files = math.ceil(ratio * len(image_files))
    sampled_file_index = np.random.choice(total_files, sample_num_files)
    sampled_image_files = np.array(image_files)[sampled_file_index]
    sampled_anno_files = np.array(anno_files)[sampled_file_index]
    assert len(sampled_anno_files) == len(sampled_image_files), "Sampled images and annotations have different lengths"

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    if not os.path.exists(save_anno_dir):
        os.makedirs(save_anno_dir)

    for f in sampled_image_files:
        f_path = os.path.join(image_dir, f)
        shutil.copy(f_path, save_img_dir)

    for f in sampled_anno_files:
        f_path = os.path.join(anno_dir, f)
        shutil.copy(f_path, save_anno_dir) 

if __name__ == "__main__":
    image_dir =  'data/test_fashion_data/image'
    anno_dir = 'data/test_fashion_data/anno'
    ratio = 0.1
    save_img_dir = 'data/test_fashion_data/sample_image'
    save_anno_dir =  'data/test_fashion_data/sample_anno'
    sampler(image_dir, anno_dir, ratio, save_img_dir, save_anno_dir)     