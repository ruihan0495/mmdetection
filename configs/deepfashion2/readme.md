This repositpry contains the code and model configs of my thesis "Efficient and Better Clothes Segmentation on DeepFashion2". The implementation of EfficientNet is borrowed from [SweetyTian][https://github.com/SweetyTian/efficientdet] with pretrained weights by [rwightman][https://github.com/rwightman/gen-efficientnet-pytorch].
## Dataset
The dataset we used in this thesis is [DeepFashion2][https://github.com/switchablenorms/DeepFashion2]. If you would like to use the dataset, please contact the authors and they will provide the password to extract the dataset. The format of dataset directory is listed below
* data/
    * DeepFashion2/
        * train/
            * image/
            * annos/
        * deep_val/
            * validation/image/
            * validation/annos/
If you want to train on the subsampled data, please run
`python tools/sampler.py`
and it will automatically generate the sampled train and validation dataset under `data/DeepFashion2/sample_train` or, with some modifications,  `data/DeepFashion2/sample_val`. Finally, to generate Coco-stype annotations, please check [here][https://github.com/switchablenorms/DeepFashion2/blob/master/evaluation/deepfashion2_to_coco.py] for how to do it.
## Installation
Our implementation is based on the [MMDetection][https://github.com/open-mmlab/mmdetection] framework and please refer to [here][https://github.com/ruihan0495/mmdetection/blob/master/docs/get_started.md] for installation instructions.
## Single GPU Training
To train our Eff-twf-sepc model, simply run
 `python tools/train.py configs/deepfashion2/eff-twf-sepc.py`
To train our Cascade Pointrend model, run
 `python tools/train.py configs/deepfashion2/cascade_pointrend.py`
## Single GPU Testing
To evaluate Eff-twf-sepc, run
`python tools/test.py configs/deepfashion2/eff-twf-sepc.py work_dirs/eff-twf-sepc/epoch_12.pth --eval bbox segm`
Note that the `work_dirs/....` is the path where you save your trained `pth` files and this is specified in the model config, e.g. see `work_dir` in the model config. If you would like to only evaluate the segmentation mAP, you can instead run 
`python tools/test.py configs/deepfashion2/eff-twf-sepc.py work_dirs/eff-twf-sepc/epoch_12.pth --eval segm`
## Distributed Training and Testing
Please refer to [here][https://github.com/ruihan0495/mmdetection/tree/master/tools] for the training and testing scripts.
## Inference
The sample code for inference looks like this:
```Python
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
config = 'configs/deepfashion2/eff-twf-sepc.py'
# Setup a checkpoint file to load
checkpoint = 'work_dirs/eff_twf_sepc/epoch_12.pth'
# initialize the detector
eff_twf_sepc = init_detector(config, checkpoint, device='cuda:0')
img_name = .... # Path to the image
result = inference_detector(pointrend, img_name)
show_result_pyplot(pointrend, img_name, result, score_thr=0.4)
```
