import torch.nn as nn

class Evaluator:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, targets in self.dataloader:
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_loss += loss.item()
        print(f'Average Loss: {total_loss / len(self.dataloader)}')
 
