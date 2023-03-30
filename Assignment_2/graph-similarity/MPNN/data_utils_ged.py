r"""This script provides a simple interface to the dataset loader that
shall be used in assignment 2, benchmarking exercises.
"""
from pathlib import Path
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx

def load_ged_dataset(name):
    r"""Loads either of the ged datasets: AIDS700nef or Linux and returns
    the dataset instance.

    param `name`: str, can be "AIDS700nef" or "LINUX"
    returns instance of `GEDDataset`
    raises ValueError
    """
    if name not in ["AIDS700nef", "LINUX"]:
        raise ValueError(f"Incorrect dataset name: {name}")
    path = str(Path('../data',name).absolute())
    dataset = GEDDataset(path, name, train=True, transform=NormalizeFeatures())
    test_dataset = GEDDataset(path, name, train=False, transform=NormalizeFeatures())
    
    # print()
    # print(f'Dataset: {dataset}:')
    # print('====================')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')

    # data = dataset[0]  # Get the first graph object.

    # print()
    # print(data)
    # print('=============================================================')

    # # Gather some statistics about the first graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    # print(f'Has self-loops: {data.has_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')
    
    # data1, data2 = dataset[0], dataset[1]
    # ged = dataset.ged[data1.i, data2.i]
    # print(ged)
    # # test data information
    # print()
    # print(f'Dataset: {test_dataset}:')
    # print('====================')
    # print(f'Number of graphs: {len(test_dataset)}')
    # print(f'Number of features: {test_dataset.num_features}')
    # print(f'Number of classes: {test_dataset.num_classes}')

    # data = test_dataset[0]  # Get the first graph object.

    # print()
    # print(data)
    # print('=============================================================')

    return dataset, test_dataset

# #converting pytorch dataset to networkx so that we can load it in dgl
# def convert_pytorch_to_networkx(dataset):
#     dataset = load_ged_dataset("LINUX")
#     nx_dataset = to_networkx(dataset)
    
    

# convert_pytorch_to_networkx("LINUX")
# convert_pytorch_to_networkx("AIDS700nef")