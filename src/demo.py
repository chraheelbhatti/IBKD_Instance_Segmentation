import torch
import matplotlib.pyplot as plt
from models.yolov10n import YOLOv10N
from config import load_config
from utils.visualization import plot_segmentation
from torchvision import transforms
from PIL import Image

def demo(config_path, image_path):
    # Load configuration
    config = load_config(config_path)

    # Initialize model
    model = YOLOv10N(config)
    model.eval()  # Set model to evaluation mode

    # Load image
    image = load_image(image_path, config['image_size'])

    # Run model on the image
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension

    # Visualize the result
    plot_segmentation(output, image_path)

def load_image(image_path, image_size):
    # Load image and transform it to tensor
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IBKD Demo")
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    parser.add_argument('--image', required=True, help='Path to the image file.')
    args = parser.parse_args()

    demo(args.config, args.image)
 
