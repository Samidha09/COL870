r"""This script provides a simple interface to the dataset loader that
shall be used in assignment 2, benchmarking exercises.
"""
from pathlib import Path
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import NormalizeFeatures


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
