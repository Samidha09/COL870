import argparse
import os.path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import f1_score
from torch_geometric.datasets import GEDDataset

import utils
from data import get_dataset
from early_stop import EarlyStopping, Stop_args
from lr import PolynomialDecayLR
from model import GED_MLP, TransformerModel


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
                        choices=['LINUX', 'AIDS700nef'])
    parser.add_argument('--path', type=str,
                        help='Path to folder containg the dataset.')
    parser.add_argument('--pyg_path', type=str,
                        help='Path to folder containg the pyg dataset.')
    parser.add_argument('--log_path', type=str, default='./',
                        help="Where to store log.csv.")
    parser.add_argument('--device', type=int, default=1, 
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=3407, 
                        help='Random seed.')
    parser.add_argument("--split", type=float, default=1.,
                    help="Fraction of training data to use for training.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top k for computing the f1 and rmse.")

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
pyg_dataset = GEDDataset(root=args.pyg_path, name=args.dataset)

graphs, idx_train, idx_val, idx_test = get_dataset(
    dataset=args.dataset,
    pe_dim=args.pe_dim,
    path=args.path,
    train_split=args.split,
)

list_processed_features = list()
for graph in graphs:
    processed_feature = utils.re_features(adj=graph[0], features=graph[1], K=args.hops)
    list_processed_features.append(processed_feature)

# model configuration
num_features = graphs[0][1].shape[1]
model = TransformerModel(hops=args.hops, 
                        n_class=args.hidden_dim, # our output is an embedding.
                        input_dim=num_features,
                        pe_dim = args.pe_dim,
                        n_layers=args.n_layers,
                        num_heads=args.n_heads,
                        hidden_dim=args.hidden_dim,
                        ffn_dim=args.ffn_dim,
                        dropout_rate=args.dropout,
                        attention_dropout_rate=args.attention_dropout).to(device)
ged_model = GED_MLP(n_layers=3, input_dim=2*args.hidden_dim, hidden_dim=128).to(device)

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

def train_valid_epoch(indices: torch.Tensor, training: bool = True):
    if training:
        model.train()
        ged_model.train()
    else:
        model.eval()
        ged_model.eval()
    loss_epoch = 0
    for idx in indices:
        src = idx
        dest = random.randint(0, len(indices))
        ged = pyg_dataset.ged[pyg_dataset[src].i, pyg_dataset[dest].i].to(device)
        features_src = list_processed_features[src].to(device)
        features_dest = list_processed_features[dest].to(device)

        if training:
            optimizer.zero_grad()
        embedding_src = model(features_src)
        embedding_dest = model(features_dest)

        pred = ged_model(torch.cat([embedding_src, embedding_dest], dim=0))

        loss_train = F.mse_loss(pred, ged)
        if training:
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
        loss_epoch += loss_train.item()

    # Normalize to get the average mse_loss
    loss_epoch /= len(indices)
    return loss_epoch

def test(indices, batch_size:int=16):
    model.eval()
    ged_model.eval()
    f1_avg = 0
    rmse_avg = 0
    # iterate over the indices
    for idx in indices:
        # choose a random batch of graphs for comparison.
        batch_indices = torch.randint(indices[0], indices[-1], (batch_size,))
        # get pred over every graph in the batch
        preds = list()
        geds = list()
        for dest in batch_indices:
            src = idx
            dest = dest.item()
            if dest >= len(list_processed_features):
                continue
            features_src = list_processed_features[src].to(device)
            features_dest = list_processed_features[dest].to(device)
            embedding_src = model(features_src)
            embedding_dest = model(features_dest)
            ged = pyg_dataset.ged[pyg_dataset[src].i, pyg_dataset[dest].i]
            pred = ged_model(torch.cat([embedding_src, embedding_dest], dim=0)).item()
            geds.append(ged)
            preds.append(pred)
        # sort the graphs based on pred. Choose top k only
        preds = torch.Tensor(preds)
        geds = torch.Tensor(geds)
        sorted_based_on_preds = batch_indices[torch.argsort(preds)][:args.top_k]
        # sort the graphs based on ged. Choose top k only
        sorted_based_on_geds = batch_indices[torch.argsort(geds)][:args.top_k]

        # compute the f1 score
        f1 = f1_score(y_true=sorted_based_on_geds, y_pred=sorted_based_on_preds, average="macro")
        # compute the rmse
        rmse =  torch.sqrt(torch.square(torch.mean(geds - preds)))
        f1_avg += f1
        rmse_avg += rmse
    f1_avg /= len(indices)
    rmse_avg /= len(indices)
    return f1_avg, rmse_avg

t_total = time.time()
stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)
df = pd.DataFrame(columns=["loss_train", "loss_val", "rmse", "f1", "time"])

start_time = time.perf_counter() # in fractional seconds.
for epoch in range(args.epochs):
    loss_train = train_valid_epoch(indices=idx_train, training=True)
    loss_val = train_valid_epoch(indices=idx_val, training=False)
    time_elapsed = time.perf_counter() - start_time

    f1_val, rmse_val = test(indices=idx_val)

    print('Epoch: {:04d}'.format(epoch+1),
        '| loss_train: {:.4f}'.format(loss_train),
        'loss_val: {:.4f}'.format(loss_val),
        '| f1_val: {:.2f}'.format(f1_val),
        '| rmse_val: {:.4f}'.format(rmse_val)
    )

    df.loc[len(df)] = [loss_train, loss_val, f1_val, rmse_val, time_elapsed]
    if early_stopping.check([loss_val], epoch):
        break

required_keys = ["split", "hops", "pe_dim", "hidden_dim", "n_layers", "top_k"]
args_for_df = {key:val for key, val in vars(args).items() if key in required_keys}
df.to_csv(f"{args.log_path}/log-{str(args_for_df)}.csv", index=False)

print("Optimization Finished!")
print("Train cost: {:.4f}s".format(time.time() - t_total))
# Restore best model
print('Loading {}th epoch'.format(early_stopping.best_epoch+1))
model.load_state_dict(early_stopping.best_state)

print(idx_test)
f1_test, rmse_test = test(indices=idx_test)
print(
    'f1_test: {:.2f}'.format(f1_val),
    '| rmse_test: {:.4f}'.format(rmse_val)
)
