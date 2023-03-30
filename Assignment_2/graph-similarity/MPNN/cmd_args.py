import argparse
import os
import torch
import time

def get_arguments():
    parser = argparse.ArgumentParser()
    #data related params
    parser.add_argument('--dataset', default='AIDS700nef', help='AIDS700nef/LINUX')
    parser.add_argument('--patience', default=30)
    parser.add_argument('--min_delta', default=10)
    # GNN params
    parser.add_argument('--saved', type=int, default=0,
                        help='1:saved, 0:unsaved')
    parser.add_argument('--in_dim', type=int, default=10,
                        help='Number of in_dim for constructed node features.')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of convolutional layers.')
    parser.add_argument('--activation', type=str, default='relu',
                        help='relu/gelu')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (between 0 and 1)')
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--batch_size",type=int,default=32)

    #others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda', help='cpu or cuda.')
    parser.add_argument('--verbose', action='store_true', help='Set to print intermediate output')

    args = parser.parse_args()
    print(args)
    return args