"""Create folders that can be supplied as --log_path and --plot_path in run.py"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", "-f", dest="folder", type=str, default="./",
                    help="Folder that will contain all the folders this script will create.")
args = parser.parse_args()

for folder in ["log", "plots"]:
    for method in ["nagphormer", "graphsage"]:
        for dataset in ["cora", "wisconsin"]:
            for exp in ["default", "training-data", "hops", "pe_dim", "n_layers", "hidden_dim"]:
                if exp in ["hops", "pe_dim"]:
                    os.system(f"mkdir -p {args.folder}/{folder}/nagphormer/{dataset}/{exp}")
                else:
                    os.system(f"mkdir -p {args.folder}/{folder}/{method}/{dataset}/{exp}")
