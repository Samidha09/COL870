import dgl
from dgl.nn import SAGEConv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, n_classes, n_layers, activation, dropout):
        super(Classifier, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                SAGEConv(hidden_dim, hidden_dim, activation=activation)
            )
        # output layer
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h, training=self.training)
            h = layer(self.g, h)
        h = self.classify(h)
        return F.softmax(h)