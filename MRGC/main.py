import sys,os; sys.path.append(os.path.abspath('./'))
import warnings
warnings.filterwarnings('ignore')
import os
from dataset import Dataset
from train import Trainer as TransTrainer

def main(args):
    dataset = Dataset(args)
    trainer = TransTrainer(dataset, args, device=args.device)
    trainer.train()

import json
import argparse
from easydict import EasyDict
def get_args():
    # ==================== Get ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--log_index', type=int)
    parser.add_argument('--reduction_rate', type=float)
    parser.add_argument('--feature_attack', type=int, default=1)
    parser.add_argument('--structure_attack', type=int, default=1)
    parser.add_argument('--label_attack', type=int, default=1)
    parser.add_argument('--feature_budget', type=float, default=0.05)
    parser.add_argument('--structure_budget', type=float, default=0.05)
    parser.add_argument('--label_budget', type=float, default=0.05)
    parser.add_argument('--transductive', type=int, default=1)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parsed_args = parser.parse_args()
    args = EasyDict(vars(parsed_args))
    # ==================== Load ====================
    json_file = f'./config/{args.name}.json'
    with open(json_file, 'r') as f:
        add_args = json.load(f)
        add_args = add_args[f'r-{args.reduction_rate}-{args.feature_budget}-{args.structure_budget}-{args.label_budget}']
    args.update(add_args)
    return args


if __name__ == "__main__":
    args = get_args()
    # args.log_file = f'./result/{args.name}-{args.reduction_rate}-{args.log_index}-{args.feature_budget}-{args.structure_budget}-{args.label_budget}.txt'
    # with open(args.log_file, 'a+') as f:
    #     f.write(f'====================Ratio {args.reduction_rate}====================\n')
    #     f.write(f'{str(args)}\n')
    print(args)
    main(args)
