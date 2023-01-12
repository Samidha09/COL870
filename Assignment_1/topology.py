import torch
from torch.nn import Linear
import torch.nn.functional as F
from data_utils import load_data
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from model import *
import argparse
import copy
import tqdm
import os

isExist = os.path.exists('./topology_generated')
if not isExist:
   os.makedirs('topology_generated')

#load data
dataset_name = 'Cora'
dataset =load_data(dataset_name)
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
data = dataset[0]  # Get the first graph object.
print()

#Experiments are being conducted on cora dataset only

#exp1 - only mlp vs gnn
args_mlp = {
'layers':2,
'hidden': 32,
'learning_rate':0.0001,
'epochs':3000
}

torch.manual_seed(0)

#mlp only
model = MLP(dataset.num_features, args_mlp['layers'], args_mlp['hidden'], dataset.num_classes) 
optimizer =torch.optim.Adam(model.parameters(), lr=args_mlp['learning_rate'], weight_decay=1e-3)
loss = nn.CrossEntropyLoss()

#train
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data.x)
    loss_train = loss(output[data.train_mask], data.y[data.train_mask])
    f1_train = f1_score(torch.argmax(output[data.train_mask], dim=1), data.y[data.train_mask], average='macro')
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    model.eval()
    with torch.no_grad():
        output = model(data.x)
        loss_val = loss(output[data.val_mask], data.y[data.val_mask])
        f1_val = f1_score(torch.argmax(output[data.val_mask], dim=1), data.y[data.val_mask], average='macro')
    return loss_train, f1_train, loss_val, f1_val

#test
def test():
    model.eval()
    output = model(data.x)
    f1_test = f1_score(torch.argmax(output[data.test_mask], dim=1), data.y[data.test_mask], average='macro')
    return f1_test

best_model = None
best_valid_loss = 1000000
best_valid_f1 = 0

for epoch in range(1, 1 + args_mlp['epochs']):
    # print('Training...')
    loss_train, f1_train, loss_val, f1_val = train()

    # print('Evaluating...')
    test_result = test()

    if(best_valid_loss > loss_val):
        best_valid_loss = loss_val
        best_valid_f1 = f1_val
        best_model = copy.deepcopy(model)

    if(epoch%2== 0):
        print(f'Epoch: {epoch:02d}, '
                f'Train: {loss_train}, '
                f'Valid: {loss_val} '
                f'Train F1: {100 * f1_train:.2f}% '
                f'Valid F1: {100 * f1_val:.2f}% '
                f'Test F1: {100 * test_result:.2f}%')

#final test with best model
best_model.eval()
output = best_model(data.x)
f1_test = f1_score(torch.argmax(output[data.test_mask], dim=1), data.y[data.test_mask], average='macro')

print(f'Test Macro F1-score: {100 * f1_test:.2f}%')
print(f'Best Val Macro F1-score: {100 * best_valid_f1:.2f}%')

f = open('./topology_generated/mlp_vs_gcn.txt', 'w')
f.write(f'Test Macro F1-score (MLP): {100 * f1_test:.2f}%\n')
f.write(f'Best Val Macro F1-score (MLP): {100 * best_valid_f1:.2f}%\n')

#gnn 
arguments = {
    'hidden_dim': 32,
    'dropout': 0.2,
    'epochs': 200,
    'mlp_num_layers':1,
    'learning_rate':0.01
    }

layers = [1, 2, 3, 4, 5, 7, 10]
f1_dict = {}
best_overall_val = 0
best_overall_test = 0
best_k = 0

#define optimizer
for k in layers:
    print(f'----------- Num GNN layers {k} --------------')
    model = GCNConv(data.num_features, arguments['hidden_dim'], dataset.num_classes, k, arguments['dropout'], arguments['mlp_num_layers'])
    optimizer =torch.optim.Adam(model.parameters(), lr=arguments['learning_rate'], weight_decay=1e-5)
    loss = nn.CrossEntropyLoss()

    #train
    def train():
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss_train = loss(output[data.train_mask], data.y[data.train_mask])
        f1_train = f1_score(torch.argmax(output[data.train_mask], dim=1), data.y[data.train_mask], average='macro')
        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        model.eval()
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            loss_val = loss(output[data.val_mask], data.y[data.val_mask])
            f1_val = f1_score(torch.argmax(output[data.val_mask], dim=1), data.y[data.val_mask], average='macro')
            
        return loss_train, f1_train, loss_val, f1_val

    #test
    def test():
        model.eval()
        output = model(data.x, data.edge_index)
        f1_test = f1_score(torch.argmax(output[data.test_mask], dim=1), data.y[data.test_mask], average='macro')
        return f1_test

    best_model = None
    best_valid_loss = 100000
    best_valid_f1 = 0

    for epoch in range(1, 1 + arguments['epochs']):
        loss_train, f1_train, loss_val, f1_val = train()
        test_result = test()

        if(best_valid_loss > loss_val):
            best_valid_loss = loss_val
            best_valid_f1 = f1_val
            best_model = copy.deepcopy(model)

        if(epoch%10== 0):
            print(f'Epoch: {epoch:02d}, '
                    f'Train: {loss_train}, '
                    f'Valid: {loss_val} '
                    f'Train F1: {100 * f1_train:.2f}% '
                    f'Valid F1: {100 * f1_val:.2f}% '
                    f'Test F1: {100 * test_result:.2f}%')

    #final test with best model
    best_model.eval()
    output = best_model(data.x, data.edge_index)
    f1_test = f1_score(torch.argmax(output[data.test_mask], dim=1), data.y[data.test_mask], average='macro')
    f1_dict[k] = f1_test

    if(best_valid_f1 > best_overall_val):
        best_overall_val = best_valid_f1
        best_overall_test = f1_test
        best_k = k

#plot
plt.plot(f1_dict.keys(), [f1_dict[key] for key in f1_dict.keys()], label = "test macro-f1")
plt.xlabel('Num GNN layers')
plt.ylabel('Macro-F1 score')
plt.title('Num GNN Layers vs Performance')
plt.legend()
plt.savefig('./topology_generated/num_layers_vs_performance.png')
plt.close()

#write results to file
f.write(f'Test Macro F1-score (GCN): {100 * best_overall_test:.2f}%\n')
f.write(f'Best Val Macro F1-score (GCN): {100 * best_overall_val:.2f}%')
f.close()

print('Best num layers hyperparameter: ', best_k)
print(f'Test Macro F1-score (GCN): {100 * best_overall_test:.2f}%\n')
print(f'Best Val Macro F1-score (GCN): {100 * best_overall_val:.2f}%')