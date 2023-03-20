"""Generate plots from dataframe for benchmarking."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set(rc={'figure.figsize': (12, 6)})

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="dataframe", type=str, help="path to dataframe.")
parser.add_argument("-o", dest="output", type=str, help="where to save the plots.")
args = parser.parse_args()

df = pd.read_csv(args.dataframe)

#* >>> f1 vs epoch
plt.figure()
xticks = list(range(len(df)))[::5]
yticks = np.arange(0, 1.1, 0.1)
plot = sns.lineplot(x=xticks, y=df["f1"][::5], marker="o")
__ = plot.set(
    xlabel="Epochs",
    ylabel="Macro f1-score",
    title="Macro f1-score vs epochs",
    # xticks=xticks,
    yticks=yticks,
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/f1-vs-epoch.png")

#* >>> f1 vs time
plt.figure()
xticks = df["time"][::5]
plot = sns.lineplot(x=xticks, y=df["f1"][::5], marker="o")
__ = plot.set(
    xlabel="Training time (seconds)",
    ylabel="Macro f1-score",
    title="Macro f1-score vs Training time",
    # xticks=df["time"][::5].round(2),
    yticks=np.arange(0, 1.1, 0.1),
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/f1-vs-time.png")

#* >>> Loss vs Epochs
plt.figure()
xticks = list(range(len(df)))[::5]
loss_vs_epochs = sns.lineplot(x=xticks, y=df["loss_train"][::5], marker="o", label="Train")
plot = sns.lineplot(x=xticks, y=df["loss_val"][::5], marker="o", label="Val")
__ = plot.set(
    xlabel="Epochs",
    ylabel="Loss",
    title="Loss vs epochs",
    # xticks=xticks,
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/loss-vs-epoch.png")
