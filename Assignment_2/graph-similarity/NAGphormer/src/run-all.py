"""
Run all the experiment. This script is only to be used
when the recommended directory strucutre is followed.
"""
import argparse
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=1, help="Device cuda ID.")
args = parser.parse_args()

for experiment in ["default", "one_hot", "training-data", "hops", "pe_dim", "n_layers", "hidden_dim", "top_k"]:
    for dataset in ["LINUX", "AIDS700nef"]:
        print(); print("#" * 50); print()
        print(f"Experiment: {experiment}, Dataset: {dataset}")
        call(
            args=[
                "python",
                "src/run.py",
                f"-e={experiment}",
                f"-d={dataset}",
                f"-p=data/nagphormer",
                f"--pyg_path=data/pyg/{dataset}",
                f"-lp=log/nagphormer/{dataset}/{experiment}",
                f"-pp=plots/nagphormer/{dataset}/{experiment}",
                f"--device={args.device}"
            ]
        )
