from .coco import COCODataset
from .voc import VOCDataset
from .cityscapes import CityscapesDataset

def get_dataset(name):
    if name == 'coco':
        return COCODataset()
    elif name == 'voc':
        return VOCDataset()
    elif name == 'cityscapes':
        return CityscapesDataset()
    else:
        raise ValueError("Unknown dataset name: {}".format(name))
 
