
import torch
import torch
from torch.nn import Linear
import torch.nn.functional as F
from data_utils import load_data
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="name of the dataset",
                    type=str)
parser.add_argument("--k", help="number of GNN layers",
                    type=int)
args = parser.parse_args()
dataset =load_data(args.dataset)
print()
print(f'Dataset: {dataset}:')
print(' Number of GNN layers ', args.k)
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print()
print(data)
print('===========================================================================================================')

# Stats about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')

