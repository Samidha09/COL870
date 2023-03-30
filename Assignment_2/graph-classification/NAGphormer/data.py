import utils
import dgl
import torch
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
import scipy.sparse as sp
from torch.utils.data import DataLoader
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, TUDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import random
import numpy as np
import math

def get_dataset(dataset, pe_dim):
    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:

        file_path = "dataset/"+dataset+".pt"

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
        # print("Adj1 : ", adj.shape)


    elif dataset in {"aminer", "reddit", "Amazon2M"}:

 
        file_path = './dataset/'+dataset+'.pt'

        data_list = torch.load(file_path)

        #adj, features, labels, idx_train, idx_val, idx_test

        adj = data_list[0]
        print("Adj1 : ", adj.shape)
        #print(type(adj))
        features = torch.tensor(data_list[1], dtype=torch.float32)
        labels = torch.tensor(data_list[2])
        idx_train = torch.tensor(data_list[3])
        idx_val = torch.tensor(data_list[4])
        idx_test = torch.tensor(data_list[5])

        graph = dgl.from_scipy(adj)


        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
       
        features = torch.cat((features, lpe), dim=1)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

        labels = torch.argmax(labels, -1)
        
        print("Adj2 : ", adj.shape)
    
    elif dataset == 'ogbg-ppa' or dataset == 'ogbg-molhiv':
        graphs = DglGraphPropPredDataset(name = dataset)
        split_idx = graphs.get_idx_split()
    
        train_loader = DataLoader(graphs[split_idx["train"][:1000]], batch_size=1, shuffle=True, collate_fn=collate_dgl)
        valid_loader = DataLoader(graphs[split_idx["valid"]], batch_size=1, shuffle=False, collate_fn=collate_dgl)
        test_loader = DataLoader(graphs[split_idx["test"]], batch_size=1, shuffle=False, collate_fn=collate_dgl)
        # print(graph)
        # print(graph.ndata['feat'].shape, graph.adj().shape, label)
        graph_list = []
        val_graphs = []
        test_graphs = []
        # for idx in split_idx["train"][:10]:
        for graph, label in train_loader:
            # print(dataset[idx])
            # graph, label = dataset[idx]
            adj = graph.adj()
            print(graph.adj().shape)
            try: 
                lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
            except:
                continue
            features = torch.cat((graph.ndata['feat'], lpe), dim=1)
            n = graph.num_nodes()  
            
            labels = torch.tensor(np.full(n, label, dtype=int)) 
            graph_list.append([adj, features, labels])
        
       
        for graph, label in valid_loader:
            adj = graph.adj()
            try: 
                lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
            except:
                continue
            features = torch.cat((graph.ndata['feat'], lpe), dim=1)
            n = graph.num_nodes()  
            labels = torch.tensor(np.full(n, label, dtype=int)) 
            val_graphs.append([adj, features, labels])
        
        for graph, label in test_loader:
            adj = graph.adj()
            try: 
                lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
            except:
                continue
            features = torch.cat((graph.ndata['feat'], lpe), dim=1)
            n = graph.num_nodes()  
            labels = torch.tensor(np.full(n, label, dtype=int)) 
            test_graphs.append([adj, features, labels])
            
        return graph_list, val_graphs, test_graphs
    
    elif dataset == 'mutag':
        data = TUDataset('MUTAG', raw_dir='./dataset')
        dataloader = dgl.dataloading.GraphDataLoader(data, batch_size=1)
        graph_list = []
        for graph, label in dataloader:
            adj = graph.adj()
            lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
            features = torch.cat((graph.ndata['node_labels'], lpe), dim=1)
            n = graph.num_nodes()
            # idx = np.arange(0, n)
            # np.random.shuffle(idx)
            # n_train, n_val, n_test = (
            #     int(n * 0.8),
            #     math.ceil(n * 0.1),
            #     math.ceil(n * 0.1),
            # )
            # # print(n_train, n_val, n_test)
            # idx_train, idx_val, idx_test = idx[0:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
            if(label.item() == 0):
                labels = torch.zeros(n)
            else:
                labels = torch.ones(n)
            graph_list.append([adj, features, labels])
            # if(len(graph_list) == 31):
            #     return graph_list
        return graph_list
    
    return adj, features, labels, idx_train, idx_val, idx_test




