import os
import torch
import torch.nn as nn
import urllib.request
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

class Mask2FormerAdapted(nn.Module):
    def __init__(self, config):
        super(Mask2FormerAdapted, self).__init__()

        # Path to the configuration file for Mask2Former
        config_url = "https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml"
        config_path = config.get("config_path", "pretrained/mask2former_coco_swin_config.yaml")

        # Download the configuration file if not found locally
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            print(f"Downloading configuration file from {config_url}...")
            urllib.request.urlretrieve(config_url, config_path)
            print(f"Configuration file saved to {config_path}.")

        # Load the configuration and initialize the model
        cfg = get_cfg()
        cfg.merge_from_file(config_path)

        # Define the path to pretrained weights
        pretrained_url = "https://huggingface.co/facebook/mask2former-swin-tiny-coco-instance/resolve/main/pytorch_model.bin"
        pretrained_path = config.get("pretrained_weights", "pretrained/mask2former_swin_tiny_coco_instance.pth")

        # Download pretrained weights if not found locally
        if not os.path.exists(pretrained_path):
            print(f"Downloading pretrained weights from {pretrained_url}...")
            urllib.request.urlretrieve(pretrained_url, pretrained_path)
            print(f"Pretrained weights saved to {pretrained_path}.")

        # Load the model with the configuration
        self.model = build_model(cfg)
        DetectionCheckpointer(self.model).load(pretrained_path)

        # Set model to evaluation mode
        self.model.eval()

    def forward(self, images):
        return self.model(images)
