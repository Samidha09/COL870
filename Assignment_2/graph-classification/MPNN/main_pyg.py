import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
from early_stop import EarlyStopper

from sklearn.metrics import f1_score
### importing OGB
from matplotlib.ticker import FormatStrFormatter
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import matplotlib.pyplot as plt
import pickle

# Training settings
parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
parser.add_argument('--device', type=int, default=1,
                    help='which gpu to use if any (default: 1)')
parser.add_argument('--gnn', type=str, default='gin-virtual',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                    help='dataset name (default: ogbg-ppa)')

parser.add_argument('--filename', type=str, default="",
                    help='filename to output result (default: )')
args = parser.parse_args()

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
multicls_criterion = torch.nn.CrossEntropyLoss()    

print(torch.cuda.is_available())

def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            total_loss += loss
            loss.backward()
            optimizer.step()
    return total_loss/len(loader.dataset) 

def validate_epoch(model, device, loader, optimizer):
    model.eval()
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            total_loss += loss
            loss.backward()
            optimizer.step()
    return total_loss/len(loader.dataset) 

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    
    return f1_score(y_true, y_pred, average='macro'), evaluator.eval(input_dict)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def main():
    ### early stop 
    early_stopper = EarlyStopper(patience=30, min_delta=10)
    ### automatic dataloading and splitting

    if(args.dataset == 'ogbg-ppa'): 
        dataset = PygGraphPropPredDataset(name = args.dataset, transform = add_zeros, root='./dataset')
    else:
        dataset = PygGraphPropPredDataset(name = args.dataset, root='./dataset')

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        
    if args.gnn == 'gin':
        model = GNN(dataset=args.dataset,gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(dataset=args.dataset,gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(dataset=args.dataset,gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'sage':
        model = GNN(dataset=args.dataset, gnn_type = 'sage', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(dataset=args.dataset,gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # valid_curve = []
    # test_curve = []
    # train_curve = []
    
    train_loss_dict = {}
    val_loss_dict = {}
    perf_dict = {}
    total_train_time = 0 

    file_train_loss = open(f"./metrics/{args.dataset}_loss_train.pkl", "wb")
    file_val_loss = open(f"./metrics/{args.dataset}_loss_val.pkl", "wb")
    file_perf = open(f"./metrics/{args.dataset}_perf.pkl", "wb")

    min_loss = 1000000
    TEST_F1 = 0
    TEST_ACC = 0
    best_model = None
    start=time.time()

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        epoch_st1 = time.time()
        train_loss = train(model, device, train_loader, optimizer)
        total_train_time += (time.time() - epoch_st1)
        print('Validating...')
        val_loss = validate_epoch(model, device, valid_loader, optimizer)
        # train_f1, train_perf = eval(model, device, train_loader, evaluator)
        # val_f1, valid_perf = eval(model, device, valid_loader, evaluator)

        print({'Train': train_loss, 'Validation': val_loss})

        # train_curve.append(train_perf[dataset.eval_metric])
        # valid_curve.append(valid_perf[dataset.eval_metric])
        # test_curve.append(test_perf[dataset.eval_metric])
        # train_curve.append(train_loss)
        # valid_curve.append(val_loss)
        # test_curve.append(test_perf[test_f1])
        
        train_loss_dict[epoch] = [train_loss.item(), total_train_time]
        val_loss_dict[epoch] = [val_loss.item(), total_train_time]
    
        if(epoch%10 == 0):
            print('Evaluating...')
            test_f1, test_perf = eval(model, device, test_loader, evaluator)
            perf_dict[epoch] = [test_perf, test_f1, total_train_time]
            
        if(val_loss.item() < min_loss):
            min_loss = val_loss.item()
            TEST_F1, TEST_ACC  = eval(model, device, test_loader, evaluator)
            best_model = model
            
        pickle.dump(train_loss_dict, file_train_loss)
        pickle.dump(val_loss_dict, file_val_loss)
        pickle.dump(perf_dict, file_perf)
        
        if early_stopper.early_stop(val_loss):             
            break
    
    print('Finished training!')
    print('Best validation loss: {}'.format(min_loss))
    print('Test score at best val loss: {}, {}'.format(TEST_F1, TEST_ACC ))
    torch.save(best_model, 'best_model.pt')    
    return train_loss_dict, val_loss_dict, perf_dict

def make_plots(train_loss_dict, val_loss_dict, perf_dict):  
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
    
if __name__ == "__main__":
    train_loss_dict, val_loss_dict, perf_dict = main()
    make_plots(train_loss_dict, val_loss_dict, perf_dict)