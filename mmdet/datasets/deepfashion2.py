from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DeepFashion2Dataset(CocoDataset):

    CLASSES = ('short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest',
                'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress',
                'sling_dress')

@DATASETS.register_module()
class DeepFashion2DatasetCP(CocoDatasetCP):

    CLASSES = ('short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest',
                'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress',
                'sling_dress')

