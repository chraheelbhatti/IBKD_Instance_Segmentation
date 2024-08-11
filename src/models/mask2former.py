import torch.nn as nn
from Mask2Former.mask2former import Mask2Former

class Mask2FormerAdapted(nn.Module):
    def __init__(self, config):
        super(Mask2FormerAdapted, self).__init__()
        # Load the Mask2Former architecture with required adaptations
        self.model = Mask2Former(config)

    def forward(self, images):
        # Adapt the forward pass to match our framework’s requirements
        return self.model(images)
 
