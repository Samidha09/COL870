import os.path
import random

import dgl
import scipy.sparse as sp
import torch
import utils
from dgl.data import (AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,
                      CiteseerGraphDataset, CoauthorCSDataset,
                      CoauthorPhysicsDataset, CoraFullDataset,
                      CoraGraphDataset, PubmedGraphDataset)
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split(data, labels, train_split: float = 0.9):
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_split, random_state=7)
    for train, test in splitter.split(data, labels):
        train = train
        test = test
    return train, test

def get_dataset(dataset, path, pe_dim, train_split: float = 1.):
    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics", "citeseer"}:
        file_path = path + dataset + ".pt"
        data_list = torch.load(file_path)

        # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "corafull":
            graph = CoraFullDataset()[0]
        elif dataset == "computer":
            graph = AmazonCoBuyComputerDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        elif dataset == "physics":
            graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]

        graph = dgl.to_bidirected(graph)
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
        features = torch.cat((features, lpe), dim=1)

    else:
        file_path = f"{path}/{dataset}.pt"
        assert os.path.exists(file_path), f"{file_path} does not exist at path."

        data_list = torch.load(file_path)
        #adj, features, labels, idx_train, idx_val, idx_test
        adj = data_list[0]

        #print(type(adj))
        features = data_list[1]
        labels = data_list[2]
        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        graph = dgl.from_scipy(adj)
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
        features = torch.cat((features, lpe), dim=1)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
        labels = torch.argmax(labels, -1)

        if train_split != 1.:
            idx_train, __ = stratified_split(
                data=idx_train,
                labels=labels[idx_train],
                train_split=train_split,
            )
    return adj, features, labels, idx_train, idx_val, idx_test
