#!/bin/bash

python src/main.py --task test --config experiments/config_coco.json
python src/main.py --task test --config experiments/config_voc.json
python src/main.py --task test --config experiments/config_cityscapes.json
 
