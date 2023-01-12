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

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="name of the dataset",
                    type=str)
parser.add_argument("--k", help="number of GNN layers",
                    type=int)
args = parser.parse_args()
dataset =load_data(args.dataset)
print()

print(f'Dataset: {dataset}:')
print(' Number of GNN layers ', args.k)
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print()
print(data)
print('===========================================================================================================')

# Stats about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')

if torch.cuda.is_available(): 
    device = "cuda:0" 
else: 
    device = "cpu"

#make model object
if(args.dataset == 'Cora'):
    arguments = {
    'seed': 0,
    'hidden_dim': 32,
    'dropout': 0.2,
    'epochs': 2000,
    'mlp_num_layers':1,
    'learning_rate':0.0001
    }
else:
    arguments = {
    'seed': 123456,
    'hidden_dim': 32,
    'dropout': 0.5,
    'epochs': 2000,
    'mlp_num_layers':1,
    'learning_rate':0.0001
    }

#define optimizer
torch.manual_seed(arguments['seed'])

model = GCNConv(data.num_features, arguments['hidden_dim'],
              dataset.num_classes, args.k,
              arguments['dropout'], arguments['mlp_num_layers'])

optimizer =torch.optim.Adam(model.parameters(), lr=arguments['learning_rate'], weight_decay=1e-5)
loss = nn.CrossEntropyLoss()

#train
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    # print("Train: ", torch.argmax(output[data.train_mask], dim=1), data.y[data.train_mask])
    loss_train = loss(output[data.train_mask], data.y[data.train_mask])
    f1_train = f1_score(torch.argmax(output[data.train_mask], dim=1), data.y[data.train_mask], average='macro')
    # f1_train = accuracy_score(torch.argmax(output[data.train_mask], dim=1), data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        loss_val = loss(output[data.val_mask], data.y[data.val_mask])
        f1_val = f1_score(torch.argmax(output[data.val_mask], dim=1), data.y[data.val_mask], average='macro')
        # f1_val = accuracy_score(torch.argmax(output[data.val_mask], dim=1), data.y[data.val_mask])
    # print(loss_train, f1_train, loss_val, f1_val)
    return loss_train, f1_train, loss_val, f1_val

#test
def test():
    model.eval()
    output = model(data.x, data.edge_index)
    loss_test = loss(output[data.test_mask], data.y[data.test_mask])
    f1_test = f1_score(torch.argmax(output[data.test_mask], dim=1), data.y[data.test_mask], average='macro')
    # f1_test = accuracy_score(torch.argmax(output[data.test_mask], dim=1), data.y[data.test_mask])
    return f1_test

best_model = None
best_valid_loss = 100000
best_valid_f1 = 0
f1_dict = {}
loss_dict = {}

for epoch in range(1, 1 + arguments['epochs']):
    # print('Training...')
    loss_train, f1_train, loss_val, f1_val = train()

    # print('Evaluating...')
    test_result = test()

    if(epoch%20 == 0):
        f1_dict[epoch] = [f1_train, f1_val]
        loss_dict[epoch] = [loss_train, loss_val]

    if(best_valid_loss > loss_val):
        best_valid_loss = loss_val
        best_model = copy.deepcopy(model)
    
    if(best_valid_f1 < f1_val):
        best_valid_f1 = f1_val

    if(epoch%2== 0):
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
# f1_test = accuracy_score(torch.argmax(output[data.test_mask], dim=1), data.y[data.test_mask])

#plots
plt.plot(f1_dict.keys(), [f1_dict[key][0] for key in f1_dict.keys()], label = "train macro-f1")
plt.plot(f1_dict.keys(), [f1_dict[key][1] for key in f1_dict.keys()], label = "val macro-f1")

plt.xlabel('Epoch')
plt.ylabel('Macro-F1 score')
plt.title('Performance (Macro-F1 score) vs Epoch')
# plt.grid(True)
plt.legend()
plt.savefig(f'2020CSY7575-train-val-{args.dataset}-{args.k}-perf.png')
plt.close()

# ----------------------------------
plt.plot(loss_dict.keys(), [loss_dict[key][0] for key in loss_dict.keys()], label = "train loss")
plt.plot(loss_dict.keys(), [loss_dict[key][1] for key in loss_dict.keys()], label = "val loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
# plt.grid(True)
plt.legend()
plt.savefig(f'2020CSY7575-train-val-{args.dataset}-{args.k}-loss.png')

# ----------------------------------
f = open(f'2020CSY7575-{args.dataset}-{args.k}-results.txt', 'w')
f.write(f'Test Macro F1-score: {100 * f1_test:.2f}%\n')
f.write(f'Best Val Macro F1-score: {100 * best_valid_f1:.2f}%')
f.close()


