import argparse
from os.path import exists

from scipy.sparse import csr_matrix
from torch import save, zeros
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.utils import to_dense_adj

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest="dataset", type=str, choices=["cora", "wisconsin"],
                    help="Name (lowercase) of the dataset.")
parser.add_argument("-i", dest="input", type=str,
                    help="Path to input dataset. For example, if cora/ is in \
                        'folder/subfolder/cora/' then use '-i folder/subfolder/'")
parser.add_argument("-o", dest="output", type=str, help="Where to store the output dataset.")
args = parser.parse_args()

if not exists(args.input):
    print(f"No data at {args.input}")
    exit(1)

if args.dataset == "cora":
    dataset = Planetoid(root=args.input, name="cora")
else:
    dataset = WebKB(root=args.input, name="wisconsin")
data = dataset[0]

# Adjacency
dense_adj = to_dense_adj(data.edge_index).squeeze(0)
scipy_adj = csr_matrix(dense_adj)

# Features, labels
features = data.x
labels = zeros(size=(features.size(0), data.y.max() + 1))
for i, label in enumerate(data.y):
    labels[i, label] = 1

# Train, validation, and test splits.
if len(data.train_mask.size()) == 1:
    # If only one train-val-test split is provided.
    train_indices = data.train_mask.nonzero().flatten()
    val_indices = data.val_mask.nonzero().flatten()
    test_indices = data.test_mask.nonzero().flatten()
else:
    # If multiple train-val-test split are provided:
    train_indices = data.train_mask[:, 0].nonzero().flatten()
    val_indices = data.val_mask[:, 0].nonzero().flatten()
    test_indices = data.test_mask[:, 0].nonzero().flatten()

data_list = [scipy_adj, features, labels, train_indices, val_indices, test_indices]
save(data_list, f"{args.output}/{args.dataset}.pt")
