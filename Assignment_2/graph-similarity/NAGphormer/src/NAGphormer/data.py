from os.path import exists
import dgl
import torch
import utils
from dgl.data import (AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,
                      CiteseerGraphDataset, CoauthorCSDataset,
                      CoauthorPhysicsDataset, CoraFullDataset,
                      CoraGraphDataset, PubmedGraphDataset)

def get_dataset(dataset, pe_dim, path, train_split: float = 1.0):
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

        # adj, features, labels, idx_train, idx_val, idx_test

        adj = data_list[0]
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
    
    elif dataset in ["LINUX", "AIDS700nef"]:
        file_path = f"{path}/{dataset}.pt"
        assert exists(file_path), f"{file_path} does not exist at path."

        dict_ = torch.load(file_path)
        train_indices = dict_["train_indices"]
        val_indices = dict_["val_indices"]
        test_indices = dict_["test_indices"]

        graphs = list()
        dropped = 0
        total = len(dict_["data_lists"])
        for idx, data_list in enumerate(dict_["data_lists"]):
            adj = data_list[0]
            graph = dgl.from_scipy(adj)

            try:
                lpe = utils.laplacian_positional_encoding(graph, pe_dim)
            except:
                # print("Dropped:", idx)
                dropped += 1
                # Drop the graph that fails to produce a positional encoding.
                # Remove from indices.
                if idx in train_indices:
                    train_indices = train_indices[train_indices != idx]
                elif idx in val_indices:
                    val_indices = val_indices[val_indices != idx]
                elif idx in test_indices:
                    test_indices = test_indices[test_indices != idx]
                continue

            features = data_list[1]
            if features is not None:
                features = torch.cat((features, lpe), dim=1)
            else:
                features = lpe
            
            adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
            graphs.append([adj, features])

        if train_split != 1.0:
            train_indices = train_indices[:int(train_split * len(train_indices))]

        print(f"\nGraphs dropped for being too small: {dropped}/{total}")
        return graphs, train_indices, val_indices, test_indices

    return adj, features, labels, idx_train, idx_val, idx_test
