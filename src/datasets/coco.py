import torch
from torchvision import transforms

class COCODataset(torch.utils.data.Dataset):
    def __init__(self):
        # Initialize dataset, e.g., load annotations
        pass

    def __len__(self):
        # Return the total number of samples
        return 10000

    def __getitem__(self, idx):
        # Return one sample and its target (dummy implementation)
        sample = torch.zeros((3, 640, 640))
        target = torch.zeros((640, 640))
        return sample, target
 
