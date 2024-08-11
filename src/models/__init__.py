from .mask2former import Mask2FormerAdapted
from .yolov10n import YOLOv10N
from .integration import FeatureBridging

def get_model(name, config):
    if name == 'mask2former':
        return Mask2FormerAdapted(config)
    elif name == 'yolov10n':
        return YOLOv10N(config)
    else:
        raise ValueError("Unknown model name: {}".format(name))
 
