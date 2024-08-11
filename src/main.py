import argparse
from train import train
from test import test
from demo import demo

def parse_args():
    parser = argparse.ArgumentParser(description="IBKD Instance Segmentation")
    parser.add_argument('--task', choices=['train', 'test', 'demo'], required=True, help='Choose the task to execute.')
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.task == 'train':
        train(args.config)
    elif args.task == 'test':
        test(args.config)
    elif args.task == 'demo':
        demo(args.config)

if __name__ == '__main__':
    main()
 
