"""Run specific benchmarking experiments provided the recommended directory structure is followed."""
import argparse
from subprocess import run

parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", dest="method", type=str,
                    choices=["nagphormer", "graphsage"],
                    help="Which method to use.")
parser.add_argument("--experiment", "-e", dest="experiment", type=str,
                    choices=[
                        "default", "training-data",
                        "hops", "pe_dim", # exclusively for nagphormer
                        "hidden_dim", "n_layers"
                    ],
                    help="Which experiment to run.")
parser.add_argument("--dataset", "-d", dest="dataset", type=str,
                    choices=["cora", "wisconsin"],
                    help="Name of the dataset.")
parser.add_argument("--device", type=int, default=1, help="Device cuda ID.")
args = parser.parse_args()

if args.method == "graphsage" and args.experiment in ["hops", "pe_dim"]:
    print("These experiments are exclusively for nagphormer.")
    exit(1)

run(
    args=[
        "python",
        "src/run.py",
        f"-m={args.method}",
        f"-e={args.experiment}",
        f"-d={args.dataset}",
        f"-p=data/{args.method}",
        f"-lp=log/{args.method}/{args.dataset}/{args.experiment}",
        f"-pp=plots/{args.method}/{args.dataset}/{args.experiment}",
        f"--device={args.device}"
    ]
)
