# from torch_geometric.nn import SAGEConv
from layer import SAGEConv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import torch

class Classifier(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, n_layers, seed, dropout, activation):
        super(Classifier, self).__init__()
        torch.manual_seed(seed)
        self.dropout = dropout
        if(activation == 'gelu'):
            self.activation = F.gelu
        else:
            self.activation = F.relu
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(dataset.num_node_features, hidden_dim))
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))
        self.lin = nn.Linear(hidden_dim, dataset.num_classes)
        

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings 
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x, edge_index, edge_attr))
            # print(x.shape)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x
