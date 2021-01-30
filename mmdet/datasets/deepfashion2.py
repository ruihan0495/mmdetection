from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DeepFashion2Dataset(CocoDataset):

    CLASSES = ('short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest',
                'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress',
                'sling dress')

