"""Run specific benchmarking experiments provided the recommended directory structure is followed."""
import argparse
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", "-e", dest="experiment", type=str,
                    choices=[
                        "default", "one_hot",
                        "training-data", "hops", "pe_dim",
                        "hidden_dim", "n_layers", "top_k",
                    ],
                    help="Which experiment to run.")
parser.add_argument("--dataset", "-d", dest="dataset", type=str,
                    choices=["LINUX", "AIDS700nef"],
                    help="Name of the dataset.")
parser.add_argument("--device", type=int, default=1, help="Device cuda ID.")
args = parser.parse_args()

call(
    args=[
        "python",
        "src/run.py",
        f"-e={args.experiment}",
        f"-d={args.dataset}",
        f"-p=data/nagphormer",
        f"--pyg_path=data/pyg/{args.dataset}",
        f"-lp=log/nagphormer/{args.dataset}/{args.experiment}",
        f"-pp=plots/nagphormer/{args.dataset}/{args.experiment}",
        f"--device={args.device}"
    ]
)
