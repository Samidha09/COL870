import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class Classifier(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, n_classes, n_layers, activation, dropout):
        super(Classifier, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        print(activation)
        self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type='sum', activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                SAGEConv(hidden_dim, hidden_dim, aggregator_type='sum', activation=activation)
            )
        # output layer
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        
    def readout(self, embs):
        return torch.sum(embs, 1)
    
    def forward(self, graphs):
        h = graphs.ndata['x']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h, training=self.training)
            h = layer(self.g, h)
            print("layer: ", i, h.shape)
        h = self.readout(h)
        print("readout: ", h.shape)
        h = self.classify(h)
        return F.softmax(h)