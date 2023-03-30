import torch
import torch.nn.functional as F
from model import Classifier
import torch.nn as nn
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from early_stop import EarlyStopper
from cmd_args import get_arguments
from sklearn.metrics import f1_score, accuracy_score
import math
import pickle
import time
import os
args = get_arguments()

if(args.device == 'cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1" #set according to gpu avalaibility
    
torch.manual_seed(args.seed)

def add_ones(data):
    data.x = torch.zeros(data.num_nodes, args.in_dim, dtype=torch.float32)
    return data

dataset = PygGraphPropPredDataset(name = args.dataset, root = '../dataset/', transform = add_ones) 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"][:math.ceil(len(split_idx["train"])*0.01)]], batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

print('Num train graphs: ', len(train_loader.dataset))
print('Num validation graphs: ', len(valid_loader.dataset))
print('Num test graphs: ',len(test_loader.dataset))


model = Classifier(dataset, args.hidden, args.n_layers, args.seed, args.dropout, args.activation).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        # print(data.x.shape)
        out = model(data.x.to(args.device), data.edge_index.to(args.device), data.edge_attr.to(args.device), data.batch.to(args.device))  # Perform a single forward pass.
        # print(out.shape, data.y.shape, data.y.view(-1,))
        loss = criterion(out, data.y.view(-1,).to(args.device))  # Compute the loss.
        total_loss += loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        
    return total_loss/len(train_loader.dataset)   
        
def validate_epoch(loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    # correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(args.device), data.edge_index.to(args.device), data.edge_attr.to(args.device), data.batch.to(args.device))  # Perform a single forward pass.
        loss = criterion(out, data.y.view(-1,).to(args.device))  # Compute the loss.
        total_loss += loss
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # correct += int((pred == data.y.to(args.device)).sum())  # Check against ground-truth labels.
        y_true.append(data.y.detach().cpu())
        y_pred.append(pred.detach().cpu())
        
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy().reshape((y_pred.shape[0],1))
    # print("Y_PRED SHAPE: ", y_pred.shape, "Y_TRUE SHAPE: ", y_true.shape,)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return total_loss/len(loader.dataset), f1, accuracy_score(y_true, y_pred)#correct / len(loader.dataset)  # Derive ratio of correct predictions.

def test(loader):
    model.eval()
    # correct = 0
    y_true = []
    y_pred = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(args.device), data.edge_index.to(args.device), data.edge_attr.to(args.device), data.batch.to(args.device))  # Perform a single forward pass.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # correct += int((pred == data.y.to(args.device)).sum())  # Check against ground-truth labels.
        y_true.append(data.y.detach().cpu())
        y_pred.append(pred.detach().cpu())
    
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy().reshape((y_pred.shape[0],1))
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return f1, accuracy_score(y_true, y_pred)#correct / len(loader.dataset)

loss_dict = {}
perf_dict = {}
file_loss = open("loss.pkl", "wb")
file_perf = open("perf.pkl", "wb")
start=time.time()
best_loss = math.inf 

for epoch in range(1, args.epochs):
    train_loss = train()
    curr_time = time.time()
    
    if(epoch%5):
        validation_loss, val_f1, val_acc = validate_epoch(valid_loader)
        # train_acc = test(train_loader)
        test_f1, test_acc = test(test_loader)
        loss_dict[epoch] = [train_loss, validation_loss, curr_time]
        perf_dict[epoch] = [val_f1, val_acc, test_f1, test_acc, curr_time]
        if(validation_loss < best_loss):
            best_loss = validation_loss
            torch.save(model.state_dict(), './best_model.pt')
        if early_stopper.early_stop(validation_loss):             
            break
        
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {validation_loss:.4f}')
        print(f'Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}')
    
pickle.dump(loss_dict, file_loss)
pickle.dump(perf_dict, file_perf)
    
    

# evaluator = Evaluator(name = 'ogbg-ppa')
# print(evaluator.expected_input_format) 
# print(evaluator.expected_output_format)  
# # In most cases, input_dict is
# # input_dict = {"y_true": y_true, "y_pred": y_pred}
# result_dict = evaluator.eval(input_dict)