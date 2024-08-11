import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_segmentation(output, image_path):
    """Visualize the segmentation results."""
    image = plt.imread(image_path)
    segmentation = torch.sigmoid(output).squeeze().cpu().numpy()
    segmentation = (segmentation > 0.5).astype(np.uint8)

    # Plot image and segmentation mask
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmentation Mask")
    plt.imshow(segmentation, cmap='gray')
    plt.axis('off')

    plt.show()
 
