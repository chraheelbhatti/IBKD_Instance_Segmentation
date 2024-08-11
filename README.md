# IBKD: An Optimal Approach to Instance Segmentation Using Knowledge Distillation from Real-Time Scenarios

## Overview

**IBKD** is an advanced framework for real-time instance segmentation, leveraging the strengths of the Mask2Former teacher model and the YOLOv10N student model. This project utilizes knowledge distillation techniques to optimize segmentation accuracy while maintaining efficiency for real-time applications. It is well-suited for tasks such as autonomous driving, video surveillance, and augmented reality.

## Features

- **Bifocal Attention Integration (BAI)**: Merges contextual insights from Mask2Former with the efficiency of YOLOv10N.
- **Contextual Feature Bridging (CFB)**: Aligns feature spaces to ensure effective knowledge transfer.
- **Context-Aware Knowledge Distillation**: Dynamically adjusts knowledge transfer based on input complexity.
- **Temporal Equilibrium System (TES)**: Stabilizes segmentation outputs in video sequences for consistent results.

## Project Structure

IBKD_Instance_Segmentation/
│
├── src/
│ ├── main.py
│ ├── config.py
│ ├── train.py
│ ├── test.py
│ ├── demo.py
│ ├── datasets/
│ │ ├── init.py
│ │ ├── coco.py
│ │ ├── voc.py
│ │ └── cityscapes.py
│ ├── models/
│ │ ├── init.py
│ │ ├── mask2former.py
│ │ ├── yolov10n.py
│ │ ├── integration.py
│ │ └── components/
│ │ ├── attention.py
│ │ └── layers.py
│ ├── utils/
│ │ ├── init.py
│ │ ├── logger.py
│ │ ├── metrics.py
│ │ └── visualization.py
│ ├── trainers/
│ │ ├── init.py
│ │ └── trainer.py
│ └── evaluators/
│ ├── init.py
│ └── evaluator.py
│
├── experiments/
│ ├── config_coco.json
│ ├── config_voc.json
│ └── config_cityscapes.json
│
└── scripts/
├── prepare_data.py
├── train.sh
└── test.sh


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/IBKD_Instance_Segmentation.git
   cd IBKD_Instance_Segmentation

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
