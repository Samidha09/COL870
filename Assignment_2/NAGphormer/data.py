import utils
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset

def get_dataset(dataset, pe_dim, isSC, isPE):
    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics"}:

        
        # "pubmed", "corafull", "computer", "photo", "cs", "physics"



        file_path = "./dataset/de/de"+"_"+dataset+"_"+str(0)+".pt"
        data_list = torch.load(file_path)

        # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]
        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        data_file = "./dataset/"+dataset+"_"+str(pe_dim)+".pt"


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


        graph = dgl.to_bidirected(graph)



        if isPE == 1:
            
            data_file = "./dataset/"+dataset+"_"+str(pe_dim)+".pt"

            if os.path.isfile(data_file):
                lpe = torch.load(data_file)

            else:
                lpe = utils.laplacian_positional_encoding(graph, pe_dim)  # return (N, hops+1, d)
                # store the data 
                torch.save(lpe, data_file)
            
            features = torch.cat((features, lpe), dim=1)



    elif dataset in {"aminer", "reddit", "Amazon2M"}:

 
        file_path = './dataset/'+dataset+'.pt'

        data_list = torch.load(file_path)

        #adj, features, labels, idx_train, idx_val, idx_test

        adj = data_list[0]
        
        #print(type(adj))
        features = torch.tensor(data_list[1], dtype=torch.float32)
        labels = torch.tensor(data_list[2])
        idx_train = torch.tensor(data_list[3])
        idx_val = torch.tensor(data_list[4])
        idx_test = torch.tensor(data_list[5])

        graph = dgl.from_scipy(adj)

        if isPE == 1:
            
            data_file = "./dataset/"+dataset+"_"+str(pe_dim)+".pt"

            if os.path.isfile(data_file):
                lpe = torch.load(data_file)

            else:
                lpe = utils.laplacian_positional_encoding(graph, pe_dim)  # return (N, hops+1, d)
                # store the data 
                torch.save(lpe, data_file)
            
            features = torch.cat((features, lpe), dim=1)

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)


        labels = torch.argmax(labels, -1)
        


    adj = utils.torch_sparse_tensor_to_sparse_mx(adj)

    
    if isSC == 1:
        adj = adj + sp.eye(adj.shape[0])
    adj = utils.normalize_adj(adj)
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_val, idx_test


