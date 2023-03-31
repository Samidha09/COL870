"""Generate plots from dataframe for benchmarking."""
import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set(rc={'figure.figsize': (12, 6)})

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="input", type=str, help="path to folder containing log dataframe.")
parser.add_argument("-o", dest="output", type=str, help="where to save the plots.")
args = parser.parse_args()

# There will be only one file in there.
for filename in glob(args.input + "/*"):
    df = pd.read_csv(filename)
    break

#* >>> f1 vs epoch
plt.figure()
xticks = list(range(len(df)))[::5]
yticks = np.arange(0, 1.1, 0.1)
plot = sns.lineplot(x=xticks, y=df["f1"][::5], marker="o")
__ = plot.set(
    xlabel="Epochs",
    ylabel="Macro f1-score",
    title="Macro f1-score vs Epochs",
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
    yticks=np.arange(0, 1.1, 0.1),
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/f1-vs-time.png")

#* >>> loss vs epochs
plt.figure()
xticks = list(range(len(df)))[::5]
plot = sns.lineplot(x=xticks, y=df["loss_train"][::5], marker="o", label="Train")
plot = sns.lineplot(x=xticks, y=df["loss_val"][::5], marker="o", label="Val")
__ = plot.set(
    xlabel="Epochs",
    ylabel="Loss",
    title="Loss vs Epochs",
)
fig = plot.get_figure()
fig.savefig(f"{args.output}/loss-vs-epoch.png")
