from sklearn.model_selection import StratifiedShuffleSplit
from torch import zeros_like, manual_seed
from torch_geometric.datasets import Planetoid, WebKB

manual_seed(7)

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
    # In case multiple train-val-test splits are provided.
    if len(data.train_mask.size()) != 1:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    if train_split != 1.0:
        train_indices, __ = stratified_split(
            data=data.train_mask.nonzero().flatten(),
            labels=data.y[data.train_mask],
            train_split=train_split,
        )
        train_mask = zeros_like(data.val_mask)
        train_mask[train_indices] = True
        data.train_mask = train_mask
    return data
