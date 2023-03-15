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
    path = str(Path('data',name).absolute())
    dataset = GEDDataset(path, name, train=True, transform=NormalizeFeatures())
    return dataset

#converting pytorch dataset to networkx so that we can load it in dgl
def convert_pytorch_to_networkx(dataset):
    dataset = load_ged_dataset("LINUX")
    nx_dataset = to_networkx(dataset)
    
    

convert_pytorch_to_networkx("LINUX")
convert_pytorch_to_networkx("AIDS700nef")