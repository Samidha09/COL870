import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Classifier
from early_stop import EarlyStopper
from cmd_args import get_arguments
from data_utils_ged import load_ged_dataset
from torch_geometric.loader import DataLoader

args = get_arguments()
data, test_data = load_ged_dataset(args.dataset)
exit(0)
train_loader = DataLoader(data, batch_size=args.batch_size)
val_loader = DataLoader(dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_data, batch_size=args.batch_size)

model = Classifier(g = dataset, in_dim=args.in_dim, hidden_dim=args.hidden, n_classes=37, n_layers=args.n_layers, activation=args.activation, dropout=args.dropout)

opt = torch.optim.Adam(model.parameters(),lr=args.lr)
early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

#train
for epoch in range(args.epochs):
    for batched_graph, labels in train_loader:
        feats = batched_graph.nodes.data['x']
        edge_index = batched_graph.nodes.data['x']
        edge_attr = batched_graph.edges.data['feat']
        logits = model(feats, edge_index, edge_attr)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    #validate
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