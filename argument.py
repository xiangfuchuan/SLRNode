import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--num_train_per_class", type=int, default=16)
    parser.add_argument("--repeating", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--conv", type=str, default='GCNConv')
    parser.add_argument("--layers", nargs='+',type=list, default=[128])
    parser.add_argument("--dc", type=float, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1704712891)
    parser.add_argument("--use_lt", type=bool, default=True)
    parser.add_argument("--pseudo_label_num", type=int, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
