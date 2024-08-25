import torch.nn as nn
import os
import torch
import urllib.request
from yolov10.models import YOLOv10
from yolov10.utils import load_pretrained_weights

class YOLOv10N(nn.Module):
    def __init__(self, config):
        super(YOLOv10N, self).__init__()

        # Path to the pretrained YOLOv10N weights
        pretrained_url = "https://huggingface.co/jameslahm/yolov10n/resolve/main/yolov10n.pth"
        pretrained_path = config.get("pretrained_weights", "pretrained/yolov10n.pth")

        # Download pretrained weights if they don't exist
        if not os.path.exists(pretrained_path):
            print(f"Downloading YOLOv10N pretrained weights from {pretrained_url}...")
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            urllib.request.urlretrieve(pretrained_url, pretrained_path)
            print(f"Downloaded YOLOv10N pretrained weights to {pretrained_path}.")

        # Load the YOLOv10 architecture
        self.model = YOLOv10(config['image_size'])
        checkpoint = torch.load(pretrained_path, map_location="cpu")  # Adjust map_location for GPUs
        self.model.load_state_dict(checkpoint, strict=False)  # Use strict=False for partial loading

    def forward(self, images):
        return self.model(images)
