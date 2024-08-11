import torch.optim as optim
import torch.nn as nn
from utils.distillation_loss import KnowledgeDistillationLoss

class Trainer:
    def __init__(self, teacher_model, student_model, dataloader, config):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataloader = dataloader
        self.config = config
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=config['learning_rate'])
        self.criterion = KnowledgeDistillationLoss(temperature=config['temperature'])

    def train(self, epochs):
        self.teacher_model.eval()  # Ensure teacher model is in eval mode
        self.student_model.train()  # Student model in train mode
        for epoch in range(epochs):
            total_loss = 0
            for images, targets in self.dataloader:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                student_outputs = self.student_model(images)

                loss = self.criterion(student_outputs, teacher_outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(self.dataloader)}')
 
