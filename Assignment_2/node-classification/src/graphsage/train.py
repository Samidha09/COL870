import argparse
from copy import deepcopy
from time import perf_counter

import torch
from pandas import DataFrame
from torch.nn.functional import softmax

from data import get_dataset
from model import Model, cal_f1_score

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--dataset", type=str, choices=["cora", "wisconsin"])
parser.add_argument("--path", type=str, help="Where is the dataset stored.\
                    For example, if 'cora/' is stored in 'folder/subfolder', then use '--path=folder/subfolder'")
parser.add_argument("--split", type=float, help="Fraction of training data to keep.")
# Model
parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the model.")
parser.add_argument("--hidden_units", type=int, default=128, help="Hidden units in each hidden layer.")
parser.add_argument("--dropout", type=float, default=0.1)
# Training
parser.add_argument("--device", type=int, default=0, help="Cuda gpu index.")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--max_epochs", type=int, default=1000)
# Output
parser.add_argument("--log_path", type=str, default="./", help="Where to store log.csv.")

args = parser.parse_args()

data = get_dataset(name=args.dataset, root=args.path, train_split=args.split)

model = Model(
    num_features=data.x.size(1),
    num_classes=data.y.max() + 1,
    num_layers=args.num_layers,
    hidden_units=args.hidden_units,
    dropout=args.dropout,
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    out = softmax(out, dim=-1)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    out = softmax(out, dim=-1)
    loss = criterion(out[mask], data.y[mask])
    f1 = cal_f1_score(out[mask], data.y[mask], data.y.max()+1)
    return loss.item(), f1.item()

# Send over to gpu.
data = data.to(args.device)
model = model.to(args.device)
best_model = deepcopy(model).to(args.device)

# Train.
PATIENCE = 30
patience = 0
best_val_loss = torch.inf
df = DataFrame(columns=["loss_train", "loss_val", "f1", "time"])

start_time = perf_counter()
for epoch in range(args.max_epochs):
    if patience == PATIENCE:
        break
    loss_train = train()
    loss_train, f1_train = test(data.train_mask)
    loss_val, f1_val = test(data.val_mask)
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        best_model = deepcopy(model)
        patience = 0
    else:
        patience += 1
    print(
        f"Epoch: {epoch:03d}"
        f" | Train loss: {loss_train:.4f}, Val loss: {loss_val:.4f}"
        f" | Train F1: {f1_train:.2f}, Val F1: {f1_val:.2f}"
    )
    elapsed_time = perf_counter() - start_time
    df.loc[len(df)] = [loss_train, loss_val, f1_val, elapsed_time]

model = deepcopy(best_model)
loss_train, f1_train = test(data.train_mask)
loss_val, f1_val = test(data.val_mask)
loss_test, f1_test = test(data.test_mask)
print(
    "Best model:\n"
    f"Train loss: {loss_train:.4f}, Val loss: {loss_val:.4f}, Test loss: {loss_test:.4f}\n"
    f"Train F1: {f1_train:.2f}, Val F1: {f1_val:.2f}, Test F1: {f1_test:.2f}"
)
df.to_csv(args.log_path + "/log.csv", index=False)
