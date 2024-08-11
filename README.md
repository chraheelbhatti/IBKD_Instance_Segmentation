# IBKD: An Optimal Approach to Instance Segmentation Using Knowledge Distillation from Real-Time Scenarios

## 1. Overview

**IBKD** is an advanced framework designed for real-time instance segmentation by leveraging the strengths of the Mask2Former teacher model and the YOLOv10N student model. It employs knowledge distillation techniques to enhance segmentation accuracy while maintaining high efficiency for real-time applications. This project is ideal for domains such as autonomous driving, video surveillance, and augmented reality.

## 2. Features

1. **Bifocal Attention Integration (BAI)**: Combines contextual insights from Mask2Former with the efficiency of YOLOv10N.
2. **Contextual Feature Bridging (CFB)**: Aligns feature spaces to ensure effective knowledge transfer.
3. **Context-Aware Knowledge Distillation**: Adjusts knowledge transfer dynamically based on input complexity.
4. **Temporal Equilibrium System (TES)**: Stabilizes segmentation outputs across video sequences for consistent results.

## 3. Project Structure

```plaintext
IBKD_Instance_Segmentation/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── train.py
│   ├── test.py
│   ├── demo.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── coco.py
│   │   ├── voc.py
│   │   └── cityscapes.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mask2former.py
│   │   ├── yolov10n.py
│   │   ├── integration.py
│   │   └── components/
│   │       ├── attention.py
│   │       └── layers.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── evaluators/
│       ├── __init__.py
│       └── evaluator.py
│
├── experiments/
│   ├── config_coco.json
│   ├── config_voc.json
│   └── config_cityscapes.json
│
└── scripts/
    ├── prepare_data.py
    ├── train.sh
    └── test.sh
```

## 4. Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/IBKD_Instance_Segmentation.git
    cd IBKD_Instance_Segmentation
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 5. Data Preparation

4. **Prepare datasets:**

    ```bash
    python scripts/prepare_data.py
    ```

## 6. Training

5. **Train models using different datasets:**

    ```bash
    bash scripts/train.sh
    ```

## 7. Evaluation

6. **Evaluate models:**

    ```bash
    bash scripts/test.sh
    ```

## 8. Demo

7. **Run a demo with a sample image:**

    ```bash
    python src/demo.py --config experiments/config_coco.json --image path/to/image.jpg
    ```

## 9. Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss changes or improvements.

## 10. License

This project is licensed under the Author(s). Contact the Primary Author for more information.

## 11. Acknowledgments

- **Mask2Former**: Based on the implementation from facebookresearch/Mask2Former.
- **YOLOv10**: Based on the implementation from THU-MIG/yolov10.
