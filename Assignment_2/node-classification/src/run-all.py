"""
Run all the experiment. This script is only to be used
when the recommended directory strucutre is followed.
"""
import argparse
from subprocess import run

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=1, help="Device cuda ID.")
args = parser.parse_args()

for experiment in ["default", "training-data", "hops", "pe_dim", "n_layers", "hidden_dim"]:
    for method in ["nagphormer", "graphsage"]:
        if method == "graphsage" and experiment in ["hops", "pe_dim"]:
            # These experiments are exclusively for nagphormer.
            continue
        for dataset in ["cora", "wisconsin"]:
            print(); print("#" * 50); print()
            print(f"Method: {method}, Experiment: {experiment}, Dataset: {dataset}")
            run(
                args=[
                    "python",
                    "src/run.py",
                    f"-m={method}",
                    f"-e={experiment}",
                    f"-d={dataset}",
                    f"-p=data/{method}",
                    f"-lp=log/{method}/{dataset}/{experiment}",
                    f"-pp=plots/{method}/{dataset}/{experiment}",
                    f"--device={args.device}"
                ]
            )
