import dgl
import torch.nn.functional as F
from model import Classifier
import torch.nn as nn
import torch
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.graphproppred import Evaluator
from dgl.dataloading import GraphDataLoader
from early_stop import EarlyStopper
from cmd_args import get_arguments

args = get_arguments()
def collate_dgl(batch):
    # batch is a list of tuple (graph, label)
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)
    return g, labels

#load dataset
dataset = DglGraphPropPredDataset(name = args.dataset, root = '../dataset/')

split_idx = dataset.get_idx_split()
#create data loader
train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, collate_fn=collate_dgl)
valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, collate_fn=collate_dgl)
test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, collate_fn=collate_dgl)

for batched_graph, labels in train_loader:
    batched_graph[i].ndata['x'] = torch.ones(10)
    print(batched_graph.edges[0].data['feat'].shape, labels)
    break
# Only an example, 7 is the input feature size
model = Classifier(g = dataset, in_dim=dataset[0], hidden=args.hidden, n_classes=37, n_layers=args.n_layers, activation=args.activation, dropout=args.dropout)
opt = torch.optim.Adam(model.parameters())
early_stopper = EarlyStopper(patience=30, min_delta=10)

#train
for epoch in range(20):
    for batched_graph, labels in train_loader:
        feats = batched_graph.ndata['attr']
        logits = model(batched_graph, feats)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    exit(0)
    validation_loss = validate_epoch(model, validation_loader)
    if early_stopper.early_stop(validation_loss):             
        break
        
        
#test


# evaluator = Evaluator(name = 'ogbg-ppa')
# print(evaluator.expected_input_format) 
# print(evaluator.expected_output_format)  
# # In most cases, input_dict is
# # input_dict = {"y_true": y_true, "y_pred": y_pred}
# result_dict = evaluator.eval(input_dict)