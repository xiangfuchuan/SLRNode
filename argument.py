import argparse
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--num_train_per_class", type=float, default=16)

    parser.add_argument("--repeating", type=int, default=1, help="The number of repeat times.")
    parser.add_argument("--runs", type=int, default=10, help="The number of runs.")
    parser.add_argument("--test_size", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=20, help="The number of epochs.")
    parser.add_argument("--conv", type=str, default='GCNConv', help="GCNConv, SAGEConv, GATConv")
    parser.add_argument("--layers", nargs='+', default=[16],
                        help="The number of hidden units of each layer of the GCN.")
    parser.add_argument("--dc", type=float, default=20)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--seed", type=int, default=1704712891)
    parser.add_argument("--use_lt", type=bool, default=True)

    return parser.parse_known_args()[0]
