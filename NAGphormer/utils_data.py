

import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import random
import torch
import utils







def known_unknown_split(
        idx: np.ndarray, nknown: int = 1500, seed: int = 4143496719):
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx

def exclude_idx(idx: np.ndarray, idx_exclude_list):
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])

def train_stopping_split(
        idx: np.ndarray, labels: np.ndarray, ntrain_per_class: int = 20,
        nstopping: int = 500, seed: int = 2413340114):
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    for i in range(max(labels) + 1):
        train_idx_split.append(rnd_state.choice(
                idx[labels == i], ntrain_per_class, replace=False))
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
            exclude_idx(idx, [train_idx]),
            nstopping, replace=False)
    return train_idx, stopping_idx



def gen_splits(labels: np.ndarray, idx_split_args,
        test: bool = False):
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(
            all_idx, idx_split_args['nknown'])
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args['nknown']
    train_idx, stopping_idx = train_stopping_split(
            known_idx, labels[known_idx], **stopping_split_args)
    if test:
        val_idx = unknown_idx
    else:
        val_idx = exclude_idx(known_idx, [train_idx, stopping_idx])
    return train_idx, stopping_idx, val_idx


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""


    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix



def load_graph(dataname, sp_list, isSC, isNor):

    train_rate = sp_list[0]
    val_rate = sp_list[1]
    test_rate = sp_list[2]

    if dataname in {"pubmed", "corafull", "computer", "photo", "cs", "physics"}:
    # CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
        #default

        if dataname == "pubmed":
            dataset = PubmedGraphDataset()
        elif dataname == "corafull":
            dataset = CoraFullDataset()
        elif dataname == "computer":
            dataset = AmazonCoBuyComputerDataset()
        elif dataname == "photo":
            dataset = AmazonCoBuyPhotoDataset()
        elif dataname == "cs":
            dataset = CoauthorCSDataset()
        elif dataname == "physics":
            dataset = CoauthorPhysicsDataset()

        #加载数据
        g = dataset[0]
        adj = g.adj()

        if isSC:
            adj = torch_sparse_tensor_to_sparse_mx(adj)
            adj + sp.eye(adj.shape[0])
            adj = sparse_mx_to_torch_sparse_tensor(adj)

        if isNor:
            adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(torch_sparse_tensor_to_sparse_mx(adj)))


        features = g.ndata['feat']
        labels = g.ndata['label']


        node_index=list(range(0, features.shape[0]))

        
        nl_list = {}

        for node in range(0, features.shape[0]):

            if labels[node].item() not in nl_list.keys():

                nl_list[labels[node].item()] = []

            nl_list[labels[node].item()] += [node]

        train_idx = []
        val_idx = []
        test_idx = []

        nodes_sum = 0

        for cate in nl_list.keys():

            rand_cate_list = nl_list[cate]
            nodes_sum += len(rand_cate_list)
            print(len(rand_cate_list))

        print(nodes_sum)
       
        if train_rate > 0.99:

            for cate in nl_list.keys():

                rand_cate_list = nl_list[cate]

                
                train_num = train_rate
               
                val_num = val_rate
             
                test_num = test_rate
                  
                random.shuffle(rand_cate_list)
                
                cate_train_list = random.sample(rand_cate_list, train_num)

                rest_list = list(set(rand_cate_list) ^ set(cate_train_list))

                cate_val_list = random.sample(rest_list, val_num)

                rest_list = list(set(rest_list) ^ set(cate_val_list))

                #cate_test_list = random.sample(rest_list, test_num)
                cate_test_list = rest_list

                train_idx += cate_train_list
                val_idx += cate_val_list
                test_idx += cate_test_list

        else:

            for cate in nl_list.keys():

                rand_cate_list = nl_list[cate]
               
                
                train_num = round(train_rate*len(rand_cate_list))
                
                val_num = round(val_rate*len(rand_cate_list))
                
                test_num = round(test_rate*len(rand_cate_list))
                  
                random.shuffle(rand_cate_list)

                cate_train_list = random.sample(rand_cate_list, train_num)

                rest_list = list(set(rand_cate_list) ^ set(cate_train_list))

                cate_val_list = random.sample(rest_list, val_num)

                rest_list = list(set(rest_list) ^ set(cate_val_list))

                #cate_test_list = random.sample(rest_list, test_num)
                cate_test_list = rest_list

                train_idx += cate_train_list
                val_idx += cate_val_list
                test_idx += cate_test_list
    
        return adj, features, labels, train_idx, val_idx, test_idx
        



