import argparse
import os.path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd

import utils
from data import get_dataset
from early_stop import EarlyStopping, Stop_args
from lr import PolynomialDecayLR
from model import TransformerModel


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
    parser.add_argument('--path', type=str,
                        help='Path to folder containg the dataset.')
    parser.add_argument('--log_path', type=str, default='./',
                        help="Where to store log.csv.")
    parser.add_argument('--device', type=int, default=1, 
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407, 
                        help='Random seed.')
    parser.add_argument("--split", type=float, default=1.,
                    help="Fraction of training data to use for training.")

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
    parser.add_argument('--epochs', type=int, default=2000,
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
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    return parser.parse_args()

args = parse_args()
print("\n>>> train.py:", args)
device = args.device

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Load and pre-process data
adj, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.path, args.pe_dim, args.split)


processed_features = utils.re_features(adj, features, args.hops)  # return (N, hops+1, d)

num_classes = labels.max().item() + 1
labels = labels.to(device) 

batch_data_train = Data.TensorDataset(processed_features[idx_train], labels[idx_train])
batch_data_val = Data.TensorDataset(processed_features[idx_val], labels[idx_val])
batch_data_test = Data.TensorDataset(processed_features[idx_test], labels[idx_test])

train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle = True)
val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle = True)
test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle = True)


# model configuration
model = TransformerModel(hops=args.hops, 
                        n_class=labels.max().item() + 1, 
                        input_dim=features.shape[1], 
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

def train_valid_epoch(epoch):
    model.train()
    loss_train_b = 0
    acc_train_b = 0
    train_output_list = list()
    train_label_list = list()
    for _, item in enumerate(train_data_loader):
        
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        optimizer.zero_grad()
        output = model(nodes_features)
        loss_train = F.cross_entropy(output, labels)
        loss_train.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_train_b += loss_train.item()
        acc_train = utils.accuracy_batch(output, labels)
        acc_train_b += acc_train.item()
        
        train_output_list.append(output)
        train_label_list.append(labels)
 
    model.eval()
    loss_val = 0
    acc_val = 0
    val_output_list = list()
    val_label_list = list()
    for _, item in enumerate(val_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        output = model(nodes_features)
        loss_val += F.cross_entropy(output, labels).item()
        acc_val += utils.accuracy_batch(output, labels).item()
        val_output_list.append(output)
        val_label_list.append(labels)

    # Compute the macro f1-score.
    train_outputs = torch.cat(train_output_list, dim=0)
    train_labels = torch.cat(train_label_list)
    f1_train = utils.cal_f1_score(train_outputs, train_labels, num_classes)
    val_outputs = torch.cat(val_output_list, dim=0)
    val_labels = torch.cat(val_label_list)
    f1_val = utils.cal_f1_score(val_outputs, val_labels, num_classes).item()

    print('Epoch: {:04d}'.format(epoch+1),
        '| loss_train: {:.4f}'.format(loss_train_b),
        'loss_val: {:.4f}'.format(loss_val),
        '| acc_train: {:.4f}'.format(acc_train_b/len(idx_train)),
        'acc_val: {:.4f}'.format(acc_val/len(idx_val)),
        '| f1_train: {:.4f}'.format(f1_train),
        'f1_val: {:.4f}'.format(f1_val),
    )
    return loss_train_b, loss_val, acc_val, f1_val

def test():
    loss_test = 0
    acc_test = 0
    for _, item in enumerate(test_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        model.eval()

        output = model(nodes_features)
        loss_test += F.cross_entropy(output, labels).item()
        acc_test += utils.accuracy_batch(output, labels).item()

    print("Test set results:",
        "loss= {:.4f}".format(loss_test),
        "accuracy= {:.4f}".format(acc_test/len(idx_test)))


t_total = time.time()
stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)
df = pd.DataFrame(columns=["loss_train", "loss_val", "f1", "time"])

start_time = time.perf_counter() # in fractional seconds.
for epoch in range(args.epochs):
    loss_train, loss_val, acc_val, f1_val = train_valid_epoch(epoch)
    time_elapsed = time.perf_counter() - start_time
    df.loc[len(df)] = [loss_train, loss_val, f1_val, time_elapsed]
    if early_stopping.check([acc_val, loss_val], epoch):
        break
# df.to_csv(f"{args.log_path}/log{args.split}.csv", index=False)
required_keys = ["split", "hops", "pe_dim", "hidden_dim", "n_layers"]
args_for_df = {key:val for key, val in vars(args).items() if key in required_keys}
df.to_csv(f"{args.log_path}/log-{str(args_for_df)}.csv", index=False)

print("Optimization Finished!")
print("Train cost: {:.4f}s".format(time.time() - t_total))
# Restore best model
print('Loading {}th epoch'.format(early_stopping.best_epoch+1))
model.load_state_dict(early_stopping.best_state)

test()
