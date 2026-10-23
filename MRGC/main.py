import sys,os; sys.path.append(os.path.abspath('./'))
import warnings
warnings.filterwarnings('ignore')
import os
import random
import numpy as np
import torch
from dataset import Dataset
from train import Trainer as TransTrainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    dataset = Dataset(args)
    # Paper (Sec. 4.1): repeat every experiment five times, report mean and std
    test_res = []
    for run in range(args.runs):
        set_seed(args.seed + run)
        with open(args.log_file, 'a+') as f:
            f.write(f'--------------------Run {run} (seed {args.seed + run})--------------------\n')
        trainer = TransTrainer(dataset, args, device=args.device)
        best_val, best_test, best_epoch = trainer.train()
        test_res.append(best_test)
        msg = f'Run {run}: best epoch {best_epoch}, Val {best_val:.4f}, Test {best_test:.4f}'
        print(msg)
        with open(args.log_file, 'a+') as f:
            f.write(msg + '\n')
    test_res = np.array(test_res) * 100
    msg = f'Final Test Accuracy over {args.runs} runs: {test_res.mean():.2f} +/- {test_res.std():.2f}'
    print(msg)
    with open(args.log_file, 'a+') as f:
        f.write(msg + '\n')

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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=5)
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
    # Any attack turned off is an ablation: log it under result/ablation/ so collect_results.py
    # (which compares result/*.txt against the paper's all-attacks table) never mixes it in
    attacks = f"f{args.feature_attack}s{args.structure_attack}l{args.label_attack}"
    result_dir = './result' if attacks == 'f1s1l1' else './result/ablation'
    os.makedirs(result_dir, exist_ok=True)
    suffix = '' if attacks == 'f1s1l1' else f'-{attacks}'
    args.log_file = f'{result_dir}/{args.name}-{args.reduction_rate}-{args.log_index}-{args.feature_budget}-{args.structure_budget}-{args.label_budget}{suffix}.txt'
    with open(args.log_file, 'a+') as f:
        f.write(f'====================Ratio {args.reduction_rate}====================\n')
        f.write(f'{str(args)}\n')
    print(args)
    main(args)
