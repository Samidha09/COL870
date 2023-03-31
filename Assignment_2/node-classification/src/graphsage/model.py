from torch import manual_seed
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import dropout
from torch_geometric.nn import SAGEConv
from torcheval.metrics.functional import multiclass_f1_score

manual_seed(7)

def cal_f1_score(output, labels, num_classes):
    f1_score_ = multiclass_f1_score(
        output,
        target=labels,
        num_classes=num_classes,
        average="macro",
    )
    return f1_score_

class Model(Module):
    def __init__(
            self,
            num_features: int,
            num_classes: int,
            n_layers: int,
            hidden_dim: int = 128,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        kwargs = {"in_channels": hidden_dim, "out_channels": hidden_dim}
        self.sage_layers = ModuleList([SAGEConv(in_channels=num_features, out_channels=hidden_dim)])
        self.sage_layers.extend([SAGEConv(**kwargs) for __ in range(n_layers - 1)])
        self.dropout = dropout
        self.linear = Linear(hidden_dim, num_classes)
        return

    def forward(self, x, edge_index):
        for layer in self.sage_layers:
            x = layer(x, edge_index)
            x = dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return x
