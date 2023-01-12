import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

#class GCN Layer
class GraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels, mlp_num_layers):
        super().__init__(aggr='add')  
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.mlp = MLP(out_channels, mlp_num_layers, out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
        self.mlp.reset_parameters()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 3: Start propagating messages.
        out = self.propagate(edge_index, x=x)
        # Step 4: Apply a final bias vector.
        out += self.bias
        return out

    def message(self, x_j):
        return x_j

    def update(self, inputs):

        return self.mlp(inputs) 

#class MLP
class MLP(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim):
        assert num_layers >= 0 , "invalid input"
        super(MLP, self).__init__()
        layer_sizes = [input_dim] + [hidden_dim]*(num_layers-1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for i,linear_tranform in enumerate(self.layers):
            x = linear_tranform(x) #torch.clamp_min(linear_tranform(x),1e-14)
            if i!= len(self.layers) - 1:
                x = F.relu(x) 
                # x = F.dropout(x, self.drop, self.training)
        return x