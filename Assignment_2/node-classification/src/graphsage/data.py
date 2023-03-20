from sklearn.model_selection import StratifiedShuffleSplit
from torch import zeros_like
from torch_geometric.datasets import Planetoid, WebKB


def stratified_split(data, labels, train_split: float = 0.9):
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_split, random_state=7)
    for train, test in splitter.split(data, labels):
        train = train
        test = test
    return train, test

def get_dataset(name:str, root:str, train_split:float = None):
    if name == "cora":
        dataset = Planetoid(root=root, name=name)
    elif name == "wisconsin":
        dataset = WebKB(root=root, name=name)
    else:
        print("Invalid dataset!")
        exit(1)
    data = dataset[0]
    if train_split is not None:
        train_indices, __ = stratified_split(
            data=data.train_mask.nonzero().flatten(),
            labels=data.y[data.train_mask],
            test_split=train_split,
        )
        train_mask = zeros_like(data.val_mask)
        train_mask[train_indices] = True
        data.train_mask = train_mask
    return data
