import torch.nn as nn
from yolov10.models import YOLOv10
from yolov10.utils import load_pretrained_weights

class YOLOv10N(nn.Module):
    def __init__(self, config):
        super(YOLOv10N, self).__init__()
        # Load the YOLOv10 architecture
        self.model = YOLOv10(config['image_size'])
        load_pretrained_weights(self.model, config['pretrained_weights'])

    def forward(self, images):
        return self.model(images)
 
