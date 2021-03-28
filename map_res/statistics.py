import numpy as np
import json
import os
import shutil

def find_bad_examples(json_file, out_name, threshold=0.5):
    bad_image_ids = []
    with open(json_file, 'r') as f:
        temp = json.loads(f.read())
    for i in temp:
        if temp[i] < threshold:
            bad_image_ids.append(i)
    with open(out_name, 'w') as f:
        json.dump(bad_image_ids, f)

def find_common_bads(json_files):
    common_bads = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            temp = json.loads(f.read())
        common_bads.append(temp)
    b = [set(common_bads[i]) for i in range(0, len(common_bads))]
    common = set.intersection(*b)
    out_name = "common_bads.json"
    with open(out_name, 'w') as f:
        json.dump(list(common), f)

def extract_names(img_dir, save_file):
    image_names = os.listdir(img_dir)
    image_names = [img.split('_')[0] for img in image_names]
    with open(save_file, 'w') as f:
        json.dump(image_names, f)
    

if __name__ == "__main__":
    '''
    out_names = ["cascade_bad.json", "mask_bad.json", "detectoRS_bad.json"]
    json_files = ["cascade_mrcnn.json", "mask_rcnn.json", "detectoRS.json"]

    for json_file, out_name in zip(json_files, out_names):
        find_bad_examples(json_file, out_name)
    find_common_bads(out_names)
    
    sample_dir = '../../../sample_val/image'
    save_dir = '../../../bad_samples/image'
    with open('common_bads.json', 'r') as f:
        bad_examples = json.loads(f.read())
    print(len(bad_examples))
    for f in bad_examples:
        f_path = os.path.join(sample_dir, f)
        shutil.copy(f_path, save_dir)
    '''

    img_dirs = ['detect_results/cascade_mrcnn/bad',
                'detect_results/cascade_mrcnn/good',
                'detect_results/detecto/bad',
                'detect_results/detecto/good',
                'detect_results/hrnet/bad',
                'detect_results/hrnet/good']
    
    save_files = ['detect_results/cascade_mrcnn/bad_names.json',
                  'detect_results/cascade_mrcnn/good_names.json',
                  'detect_results/detecto/bad_names.json',
                  'detect_results/detecto/good_names.json',
                  'detect_results/hrnet/bad_names.json',
                  'detect_results/hrnet/good_names.json']

    for img_dir, save_file in zip(img_dirs, save_files):
        extract_names(img_dir, save_file)
