"""Convert PyG's GED dataset to a format that nagphormer supports."""
import argparse
from os.path import exists

from scipy.sparse import csr_matrix
from torch import arange, manual_seed, randperm, save
from torch_geometric.datasets import GEDDataset
from torch_geometric.utils import to_dense_adj

manual_seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest="dataset", type=str, choices=["LINUX", "AIDS700nef"],
                    help="Name (lowercase) of the dataset.")
parser.add_argument("-i", dest="input", type=str,
                    help="Path to input dataset. For example, if cora/ is in \
                        'folder/subfolder/LINUX/' then use '-i folder/subfolder/LINUX'")
parser.add_argument("-o", dest="output", type=str, help="Where to store the output dataset.")

args = parser.parse_args()
if not exists(args.input):
    print(f"No data at {args.input}.")
    exit(1)

dataset = GEDDataset(root=args.input, name=args.dataset)
data_lists = list()
for data in dataset:
    # Adjacency
    dense_adj = to_dense_adj(data.edge_index).squeeze(0)
    scipy_adj = csr_matrix(dense_adj)
    # Features
    features = None
    if "x" in dir(data):
        features = data.x
    data_list = [scipy_adj, features]
    data_lists.append(data_list)

indices = arange(len(dataset))
if args.dataset == "LINUX":
    test_graphs = 140
else:
    test_graphs = 200
train_indices = indices[:len(indices) - test_graphs]
test_indices = indices[len(indices) - test_graphs:]

val_indices = train_indices[int(0.9 * len(train_indices)):]
train_indices = train_indices[:int(0.9 * len(train_indices))]

dict_ = dict()
dict_["data_lists"] = data_lists
dict_["train_indices"] = train_indices
dict_["val_indices"] = val_indices
dict_["test_indices"] = test_indices

save(dict_, f"{args.output}/{args.dataset}.pt")
