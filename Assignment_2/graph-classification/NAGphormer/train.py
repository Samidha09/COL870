from data import get_dataset

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from matplotlib.ticker import FormatStrFormatter
from model import TransformerModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
from sklearn.metrics import f1_score
import argparse
import math
import matplotlib.pyplot as plt
import pickle

# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='pubmed',
                        help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=1, 
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407, 
                        help='Random seed.')

    # model parameters
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=30, 
                        help='Patience for early stopping')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--graph_batch_size', type=int, default=32,
                        help='Batch size')
    return parser.parse_args()

args = parse_args()



device="cuda:1"
# torch.cuda.set_device(1)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

def create_list_of_idxs(idx_train):
    train_batch_lists = []
    num_batches = math.ceil(len(idx_train)/args.graph_batch_size)
    bs = args.graph_batch_size
    for i in range(num_batches):
        if(i == num_batches-1):
            train_batch_lists.append(idx_train[i*bs: ])
        else:
            train_batch_lists.append(idx_train[i*bs: (i+1)*bs])
    
    return train_batch_lists

# Load and pre-process data
if(args.dataset == 'mutag'):
    graph_list = get_dataset(args.dataset, args.pe_dim)
    n = len(graph_list)
    idx = np.arange(0, n, dtype=int)
    np.random.shuffle(idx)
    n_train, n_val, n_test = (
        int(n * 0.8),
        math.ceil(n * 0.1),
        math.ceil(n * 0.1),
    )
    idx_train, idx_val, idx_test = idx[0:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    train_batch_lists = create_list_of_idxs(idx_train)
    val_graphs = [graph_list[i] for i in idx_val]
    test_graphs = [graph_list[i] for i in idx_test]
elif(args.dataset == 'ogbg-ppa' or args.dataset == 'ogbg-molhiv'):
    graph_list, val_graphs, test_graphs = get_dataset(args.dataset, args.pe_dim)
    idx_train = np.arange(0, len(graph_list), dtype=int)
    np.random.shuffle(idx_train)
    train_batch_lists = create_list_of_idxs(idx_train)
else:
    adj, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.pe_dim)

num_feats = graph_list[0][1].shape[1]
# model configuration
model = TransformerModel(hops=args.hops, 
                        n_class=args.num_classes, 
                        input_dim=num_feats, 
                        pe_dim = args.pe_dim,
                        n_layers=args.n_layers,
                        num_heads=args.n_heads,
                        hidden_dim=args.hidden_dim,
                        ffn_dim=args.ffn_dim,
                        dropout_rate=args.dropout,
                        attention_dropout_rate=args.attention_dropout).to(device)

print(model)
print('total params:', sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
lr_scheduler = PolynomialDecayLR(
                optimizer,
                warmup_updates=args.warmup_updates,
                tot_updates=args.tot_updates,
                lr=args.peak_lr,
                end_lr=args.end_lr,
                power=1.0,
            )

def train_one_batch(batch_ids):
    loss_batch = 0
    for k in range(len(batch_ids)):
        i = batch_ids[k]
        adj = graph_list[i][0]
        features = graph_list[i][1]
        labels = graph_list[i][2]
        label = torch.max(labels)
        # idx_train = graph_list[i][3]
        # idx_val = graph_list[i][4]
        # idx_test = graph_list[i][5]


        processed_features = utils.re_features(adj, features, args.hops)  # return (N, hops+1, d)


        labels = labels.to(device) 

        # batch_data = Data.TensorDataset(processed_features[idx_train], labels[idx_train])
        # batch_data_val = Data.TensorDataset(processed_features[idx_val], labels[idx_val])
        # batch_data_test = Data.TensorDataset(processed_features[idx_test], labels[idx_test])


        # train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle = True)
        # val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle = True)
        # test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle = True)
        
        batch_data = Data.TensorDataset(processed_features, labels)
        train_data_loader = Data.DataLoader(batch_data, batch_size=args.batch_size, shuffle = True)
        loss = train_one_graph(adj,features,label,train_data_loader)
        loss_batch+=loss
        
    return loss_batch/len(batch_ids)
        
def train_one_graph(adj,features,label,train_data_loader):  
    for _, item in enumerate(train_data_loader):
        
        nodes_features = item[0].to(device)
        labels = item[1].to(device)
        output = model(nodes_features)
        graph_emb = torch.mean(output, dim = 0).softmax(dim=-1)
        # print("Output: ", output)
        # print(graph_emb)
        # loss_train = F.cross_entropy(graph_emb, torch.tensor(label, dtype=torch.int64).to(device))#nll_loss
        loss_train = F.cross_entropy(graph_emb, label.clone().detach().to(device))
    return loss_train
    

def validate_epoch():
    loss_total=0
    correct_total=0
    for i in range(len(val_graphs)):
        adj = val_graphs[i][0]
        features = val_graphs[i][1]
        labels = val_graphs[i][2]
        label = torch.max(labels)
        # idx_train = val_graphs[i][3]
        # idx_val = val_graphs[i][4]
        # idx_test = val_graphs[i][5]


        processed_features = utils.re_features(adj, features, args.hops)  # return (N, hops+1, d)


        labels = labels.to(device) 

        batch_data = Data.TensorDataset(processed_features, labels)
        #keep batch size greater than number of nodes in max size graph
        data_loader = Data.DataLoader(batch_data, batch_size=args.batch_size, shuffle = True)
       
        loss, correct = validate_one_graph(adj,features,label,data_loader)
        loss_total+=loss
        correct_total+=correct
        
    return loss_total/len(val_graphs), correct_total/len(val_graphs)

def validate_one_graph(adj,features,label,data_loader):
    for _, item in enumerate(data_loader):   
        nodes_features = item[0].to(device)
        #not of any use ---
        labels = item[1].to(device)
        # ---------
        output = model(nodes_features)
        graph_emb = torch.mean(output, dim = 0).softmax(dim=-1)
        loss_val = F.cross_entropy(graph_emb, label.clone().detach().to(device))#nll_loss
        prediction = torch.argmax(graph_emb)
    return loss_val, (prediction == label)

def test():
    correct_total=0
    y_pred = []
    y_true = []
    model.eval()
    for i in range(len(test_graphs)):
        adj = test_graphs[i][0]
        features = test_graphs[i][1]
        labels = test_graphs[i][2]
        label = torch.max(labels)
        # idx_train = test_graphs[i][3]
        # idx_val = test_graphs[i][4]
        # idx_test = test_graphs[i][5]

        processed_features = utils.re_features(adj, features, args.hops)  # return (N, hops+1, d)

        labels = labels.to(device) 

        batch_data = Data.TensorDataset(processed_features, labels)
        #keep batch size greater than number of nodes in max size graph
        data_loader = Data.DataLoader(batch_data, batch_size=args.batch_size, shuffle = True)
       
        prediction, correct = test_one_graph(adj,features,label,data_loader)
        correct_total+=correct
        y_pred.append(prediction.cpu())
        y_true.append(label.cpu())
    
    return correct_total/len(test_graphs), f1_score(y_true, y_pred, average='macro')

def test_one_graph(adj,features,label,data_loader):
    for _, item in enumerate(data_loader):   
        nodes_features = item[0].to(device)
        labels = item[1].to(device)
        output = model(nodes_features)
        graph_emb = torch.mean(output, dim = 0).softmax(dim=-1)
        prediction = torch.argmax(graph_emb)
    
    return prediction, (prediction == label)


t_total = time.time()
stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)

train_loss_dict = {}
val_loss_dict = {}
perf_dict = {}
total_train_time = 0

file_train_loss = open(f"./metrics/{args.dataset}_loss_train.pkl", "wb")
file_val_loss = open(f"./metrics/{args.dataset}_loss_val.pkl", "wb")
file_perf = open(f"./metrics/{args.dataset}_perf.pkl", "wb")
    
for epoch in range(args.epochs):
    epoch_st1 = time.time()
    model.train()
    loss_total = 0
    for j in range(len(train_batch_lists)):
        batch_ids = train_batch_lists[j]
        optimizer.zero_grad()
        loss_batch = train_one_batch(batch_ids)
        loss_total+=loss_batch
        loss_batch.backward()
        optimizer.step()
        lr_scheduler.step()
    total_train_time += (time.time() - epoch_st1)
    
    model.eval()
    loss_val, acc_val = validate_epoch()
    
    train_loss_dict[epoch] = [loss_total.item()/len(train_batch_lists), total_train_time]
    val_loss_dict[epoch] = [loss_val.item(), acc_val.item(), total_train_time]
    
    if(epoch%10 == 0):
        acc, f1 = test()
        perf_dict[epoch] = [acc.item(), f1, total_train_time]
        
    print(f'Epoch: {epoch}, Train Loss: {loss_total/len(train_batch_lists)}, Val Loss, Acc: {loss_val} {acc_val}')
    if early_stopping.check([acc_val.detach().cpu(), loss_val.detach().cpu()], epoch):
        break
    
    pickle.dump(train_loss_dict, file_train_loss)
    pickle.dump(val_loss_dict, file_val_loss)
    pickle.dump(perf_dict, file_perf)
    
print("Optimization Finished!")
print("Train cost: {:.4f}s".format(time.time() - t_total))
# Restore best model
print('Loading {}th epoch'.format(early_stopping.best_epoch+1))
model.load_state_dict(early_stopping.best_state)

acc, f1 = test()

print(f"Final Test Acc: {acc}, F1 score: {f1}")

def make_plots():  
    #epoch vs loss  
    loss_list = [val[0] for val in train_loss_dict.values()]
    loss_list_val = [val[0] for val in val_loss_dict.values()]
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
   
    plt.plot(train_loss_dict.keys(), loss_list, label="Train")
    plt.plot(val_loss_dict.keys(), loss_list_val, label="Validation")
    plt.legend()
    filename = "./plots/"+args.dataset + 'epoch_vs_loss' + ".png"
    plt.savefig(filename)
    plt.close()

    #train time vs loss
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Loss')
    plt.plot([val[1] for val in val_loss_dict.values()], loss_list_val, label="Validation")
    plt.plot([val[1] for val in train_loss_dict.values()], loss_list, label="Train")
    plt.legend()
    filename = "./plots/"+ args.dataset + "traintime_vs_loss" + ".png"
    plt.savefig(filename)
    plt.close()

    #train_time vs performance
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test f1-Score (macro-avg)')
    #plt.plot(loss_list, label="Train")
    plt.plot([val[2] for val in perf_dict.values()], [val[1] for val in perf_dict.values()])
    #plt.ylim([0, 0.01])
    plt.legend()
    filename = "./plots/"+ args.dataset + "traintime_vs_perf" + ".png"
    plt.savefig(filename)
    plt.close()

    #epoch vs performance
    plt.xlabel('Epochs')
    plt.ylabel('Test f1-Score (macro-avg)')
    #plt.plot(loss_list, label="Train")
    plt.plot(perf_dict.keys(), [val[1] for val in perf_dict.values()])
    plt.legend()
    filename = "./plots/"+ args.dataset + "epoch_vs_perf" + ".png"
    plt.savefig(filename)
    plt.close()
    
make_plots()
# def train_valid_epoch(epoch):
    
#     model.train()
#     loss_train_b = 0
#     acc_train_b = 0
#     for _, item in enumerate(train_data_loader):
        
#         nodes_features = item[0].to(device)
#         labels = item[1].to(device)

#         optimizer.zero_grad()
#         output = model(nodes_features)
#         graph_emb = torch.mean(output, dim = 0).softmax(dim=-1)
#         print("Output: ", output)
#         print(graph_emb)
#         loss_train = F.cross_entropy(graph_emb, torch.tensor(label, dtype=torch.int64).to(device))#nll_loss
#         print(loss_train)
#         exit(0)
#         loss_train.backward()
#         optimizer.step()
#         lr_scheduler.step()

#         loss_train_b += loss_train.item()
#         acc_train = utils.accuracy_batch(output, labels)
#         acc_train_b += acc_train.item()
        
    
#     model.eval()
#     loss_val = 0
#     acc_val = 0
#     for _, item in enumerate(val_data_loader):
#         nodes_features = item[0].to(device)
#         labels = item[1].to(device)



#         output = model(nodes_features)
#         loss_val += F.nll_loss(output, labels).item()
#         acc_val += utils.accuracy_batch(output, labels).item()
        

#     print('Epoch: {:04d}'.format(epoch+1),
#         'loss_train: {:.4f}'.format(loss_train_b),
#         'acc_train: {:.4f}'.format(acc_train_b/len(idx_train)),
#         'loss_val: {:.4f}'.format(loss_val),
#         'acc_val: {:.4f}'.format(acc_val/len(idx_val)))

#     return loss_val, acc_val

# def test():

#     loss_test = 0
#     acc_test = 0
#     for _, item in enumerate(test_data_loader):
#         nodes_features = item[0].to(device)
#         labels = item[1].to(device)


#         model.eval()

#         output = model(nodes_features)
#         loss_test += F.nll_loss(output, labels).item()
#         acc_test += utils.accuracy_batch(output, labels).item()

#     print("Test set results:",
#         "loss= {:.4f}".format(loss_test),
#         "accuracy= {:.4f}".format(acc_test/len(idx_test)))


        

