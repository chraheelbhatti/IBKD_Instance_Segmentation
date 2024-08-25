import torch
from models.mask2former import Mask2FormerAdapted
from models.yolov10n import YOLOv10N # Ensure correct import path
from trainers.trainer import Trainer
from config import load_config
from datasets import get_dataset




def train(config_path):
    # Load configuration
    config = load_config(config_path)

    # Initialize dataset and dataloader
    dataset = get_dataset(config['dataset'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize models
    teacher_model = Mask2FormerAdapted(config)
    student_model = YOLOv10NAdapted(config)

    # Initialize trainer
    trainer = Trainer(teacher_model, student_model, dataloader, config)

    # Start training
    trainer.train(config['epochs'])

if __name__ == "__main__":
    config_path = "config/your_config.yaml"  # Update this with your actual config path
    train(config_path)
