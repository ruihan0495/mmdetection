'''
Randonly sample 10% of the training, val, and test data 
from DeepFashion2 dataset.
'''
import os
import numpy as np
import shutil
import math

train_image_dir = 'data/DeepFashion2/train/image'
train_anno_dir = 'data/DeepFashion2/train/annos'
train_save_image_dir = 'data/DeepFashion2/sample_train/image'
train_save_anno_dir = 'data/DeepFashion2/sample_train/annos'

val_image_dir = 'data/DeepFashion2/deep_val/validation/image'
val_anno_dir = 'data/DeepFashion2/deep_val/validation/annos'
val_save_image_dir = 'data/DeepFashion2/sample_val/image'
val_save_anno_dir = 'data/DeepFashion2/sample_val/annos'


def sampler(image_dir, anno_dir, ratio, save_img_dir, save_anno_dir):
    image_files = os.listdir(image_dir)
    anno_files = os.listdir(anno_dir)
    total_files = len(image_files)
    assert len(image_files) == len(anno_files), "Annotations and images don't match!"
    sample_num_files = math.ceil(ratio * len(image_files))
    sampled_image_files = np.random.choice(image_files, sample_num_files)
    sampled_image_prefix = [sampled_image_file.split('.')[0] for sampled_image_file in sampled_image_files]
    sampled_anno_files = [image_prefix + '.json' for image_prefix in sampled_image_prefix]
    
    #sampled_file_index = np.random.choice(total_files, sample_num_files)
    #sampled_image_files = np.array(image_files)[sampled_file_index]
    #sampled_anno_files = np.array(anno_files)[sampled_file_index]
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


def test_sample(image_dir, anno_dir, ratio):
    image_files = os.listdir(image_dir)
    anno_files = os.listdir(anno_dir)
    assert len(image_files) == len(anno_files), "Annotations and images don't match!"
    total_files = len(image_files)
    sample_num_files = math.ceil(ratio * len(image_files))
    print("totoal files {}, totoal sampled files {}".format(total_files, sample_num_files))


if __name__ == "__main__":
    image_dir =  train_image_dir
    anno_dir = train_anno_dir
    ratio = 0.1
    save_img_dir = train_save_image_dir
    save_anno_dir =  train_save_anno_dir
    sampler(image_dir, anno_dir, ratio, save_img_dir, save_anno_dir)  
    #test_sample(image_dir, anno_dir, ratio)     