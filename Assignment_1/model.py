import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import ModuleList
from layer import *

#class GCN
class GCNConv(nn.Module):
    """
    k-layer GCN used in GNN Explainer synthetic tasks, including
    """

    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, mlp_num_layers):
        super(GCNConv, self).__init__()
        self.layers = ModuleList()
        for i in range(nlayers):
            if(i==0):
                self.layers.append(GraphConvolution(nfeat, nhid, mlp_num_layers))
            else:
                self.layers.append(GraphConvolution(nhid, nhid, mlp_num_layers))
        self.lin = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x, adj))#gnn
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        return x #F.softmax(x, dim=1)

