import torch
from models.yolov10n import YOLOv10N
from evaluators.evaluator import Evaluator
from config import load_config
from datasets import get_dataset

def test(config_path):
    # Load configuration
    config = load_config(config_path)

    # Initialize dataset and dataloader
    dataset = get_dataset(config['dataset'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = YOLOv10N(config)

    # Initialize evaluator
    evaluator = Evaluator(model, dataloader, config)

    # Start evaluation
    evaluator.evaluate()
 
