import torch
from models.mask2former import Mask2FormerAdapted
from models.yolov10n import YOLOv10N
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
    student_model = YOLOv10N(config)

    # Initialize trainer
    trainer = Trainer(teacher_model, student_model, dataloader, config)

    # Start training
    trainer.train(config['epochs'])
 
