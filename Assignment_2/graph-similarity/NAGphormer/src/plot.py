"""Generate plots from dataframes with varying volume of training data."""
import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set(rc={'figure.figsize': (12, 6)})

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="input", type=str, help="path to folder containing log dataframes.")
parser.add_argument("-o", dest="output", type=str, help="where to save the plots.")
parser.add_argument("-e", dest="experiment", type=str, help="Which experiment to run.",
                    choices=["training-data",
                             "hops", "pe_dim", # exclusively for nagphormer
                             "hidden_dim", "n_layers", "top_k"])
args = parser.parse_args()
if args.experiment == "training-data":
    args.experiment = "split"

# Read all dataframes.
dataframes = dict()
for filename in glob(args.input + "/*"):
    #Extract the dictionary from the filename.
    args_file = eval(filename.split("/")[-1][4: -4]) # 'log-' onwards, upto .csv.
    dataframes[args_file[args.experiment]] = pd.read_csv(filename)

# Create dfs for f1s, train losses, val losses
max_col_len = max([len(df) for df in dataframes.values()])
df_f1 = pd.DataFrame({"temp": list(range(max_col_len))})
df_rmse = pd.DataFrame({"temp": list(range(max_col_len))})
df_loss_train = pd.DataFrame({"temp": list(range(max_col_len))})
df_loss_val = pd.DataFrame({"temp": list(range(max_col_len))})

for key in dataframes:
    df_f1[key] = dataframes[key]["f1"]
    df_rmse[key] = dataframes[key]["rmse"]
    df_loss_train[key] = dataframes[key]["loss_train"]
    df_loss_val[key] = dataframes[key]["loss_val"]

df_f1.drop("temp", axis=1, inplace=True)
df_rmse.drop("temp", axis=1, inplace=True)
df_loss_train.drop("temp", axis=1, inplace=True)
df_loss_val.drop("temp", axis=1, inplace=True)

# for key in dataframes:
#     df_f1 = df_f1.assign(key = dataframes[key]["f1"])
#     df_loss_train = df_loss_train.assign(key = dataframes[key]["loss_train"])
#     df_loss_val = df_loss_val.assign(key = dataframes[key]["loss_val"])

df_f1 = df_f1.iloc[::5]
df_rmse = df_rmse[::5]
df_loss_train = df_loss_train.iloc[::5]
df_loss_val = df_loss_val.iloc[::5]

# Generate three plots.
#* >>> f1 vs epoch
plt.figure()
xticks = list(range(len(df_f1)))
yticks = np.arange(0, 1.1, 0.1)
plot = sns.lineplot(data=df_f1, marker="o")
__ = plot.set(
    xlabel="Epochs",
    ylabel="Macro f1-score",
    title="Macro f1-score vs Epochs",
    yticks=yticks,
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/f1-vs-epoch.png")

#* >>> rmse vs epoch
plt.figure()
xticks = list(range(len(df_rmse)))
plot = sns.lineplot(data=df_rmse, marker="o")
__ = plot.set(
    xlabel="Epochs",
    ylabel="RMSE",
    title="RMSE vs Epochs",
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/rmse-vs-epoch.png")

#* >>> Training loss vs Epochs
plt.figure()
xticks = list(range(len(df_loss_train)))
plot = sns.lineplot(data=df_loss_train, marker="o")
__ = plot.set(
    xlabel="Epochs",
    ylabel="Loss",
    title="Training loss vs Epochs",
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/train-loss-vs-epoch.png")

#* >>> Validation loss vs Epochs
plt.figure()
xticks = list(range(len(df_loss_val)))
plot = sns.lineplot(data=df_loss_val, marker="o")
__ = plot.set(
    xlabel="Epochs",
    ylabel="Loss",
    title="Validation loss vs Epochs",
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/val-loss-vs-epoch.png")
