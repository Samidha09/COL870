"""Run experiments to generate log files."""
from argparse import ArgumentParser
from subprocess import run


# * >>> Arguments.
parser = ArgumentParser()
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
parser.add_argument("--path", "-p", type=str,
                    help="Path to folder containg the dataset."\
                        " For nagphormer, if 'cora.pt' is in 'folder/', use '-p=folder/'."\
                        " For graphsage, if 'cora/' directory is in 'folder/', use '-p=folder/'")
parser.add_argument("--log_path", "-lp", type=str, help="Where to store log.csv.")
parser.add_argument("--plot_path", "-pp", type=str, help="Where to save the plots.")
parser.add_argument("--device", type=int, help="Device cuda ID.")

args = parser.parse_args()
print("\n>>> run.py:", args)

if args.method == "graphsage" and args.experiment in ["hops", "pe_dim"]:
    print("These experiments are exclusively for nagphormer.")
    exit(1)

training_data_splits = [1.]
if args.experiment == "training-data":
    training_data_splits = [0.2, 0.4, 0.6, 0.8, 1.0]


# * >>> Benchmarking.
if args.experiment == "default" or args.experiment == "training-data":
    for t_split in training_data_splits:
        print(f"\n>>> Training split = {t_split}")
        if args.method == "nagphormer":
            run(
                args=[
                    "python",
                    "src/NAGphormer/train.py",
                    f"--dataset={args.dataset}",
                    f"--path={args.path}",
                    f"--log_path={args.log_path}",
                    f"--device={args.device}",
                    f"--split={t_split}",
                    "--patience=30",
                    "--batch_size=2000",
                    "--dropout=0.1",
                    "--hidden_dim=512",
                    "--hops=3",
                    "--n_heads=8",
                    "--n_layers=1",
                    "--pe_dim=3",
                    "--peak_lr=0.01",
                    "--weight_decay=1e-05",
                ]
            )

        elif args.method == "graphsage":
            run(
                args=[
                    "python",
                    "src/graphsage/train.py",
                    f"--dataset={args.dataset}",
                    f"--path={args.path}",
                    f"--log_path={args.log_path}",
                    f"--device={args.device}",
                    f"--split={t_split}",
                    "--dropout=0.1",
                    "--hidden_dim=64",
                    "--n_layers=3",
                    "--lr=1e-3",
                    "--weight_decay=1e-05",
                ]
            )

elif args.experiment == "hops":
    for hops in range(1, 10):
        run(
            args=[
                "python",
                "src/NAGphormer/train.py",
                f"--dataset={args.dataset}",
                f"--path={args.path}",
                f"--log_path={args.log_path}",
                f"--device={args.device}",
                f"--split=1.0",
                "--patience=30",
                "--batch_size=2000",
                "--dropout=0.1",
                "--hidden_dim=512",
                f"--hops={hops}",
                "--n_heads=8",
                "--n_layers=1",
                "--pe_dim=3",
                "--peak_lr=0.01",
                "--weight_decay=1e-05",
            ]
        )

elif args.experiment == "pe_dim":
    for pe_dim in [3, 15, 30, 60, 120, 240]:
        run(
            args=[
                "python",
                "src/NAGphormer/train.py",
                f"--dataset={args.dataset}",
                f"--path={args.path}",
                f"--log_path={args.log_path}",
                f"--device={args.device}",
                f"--split=1.0",
                "--patience=30",
                "--batch_size=2000",
                "--dropout=0.1",
                "--hidden_dim=512",
                f"--hops=3",
                "--n_heads=8",
                "--n_layers=1",
                f"--pe_dim={pe_dim}",
                "--peak_lr=0.01",
                "--weight_decay=1e-05",
            ]
        )

elif args.experiment == "hidden_dim":
    for hidden_dim in [32, 64, 128, 256, 512]:
        if args.method == "nagphormer":
            run(
                args=[
                    "python",
                    "src/NAGphormer/train.py",
                    f"--dataset={args.dataset}",
                    f"--path={args.path}",
                    f"--log_path={args.log_path}",
                    f"--device={args.device}",
                    f"--split=1.0",
                    "--patience=30",
                    "--batch_size=2000",
                    "--dropout=0.1",
                    f"--hidden_dim={hidden_dim}",
                    "--hops=3",
                    "--n_heads=8",
                    "--n_layers=1",
                    "--pe_dim=3",
                    "--peak_lr=0.01",
                    "--weight_decay=1e-05",
                ]
            )

        elif args.method == "graphsage":
            run(
                args=[
                    "python",
                    "src/graphsage/train.py",
                    f"--dataset={args.dataset}",
                    f"--path={args.path}",
                    f"--log_path={args.log_path}",
                    f"--device={args.device}",
                    f"--split=1.0",
                    "--dropout=0.1",
                    f"--hidden_dim={hidden_dim}",
                    "--n_layers=3",
                    "--lr=1e-3",
                    "--weight_decay=1e-05",
                ]
            )

elif args.experiment == "n_layers":
    for n_layers in range(1, 6):
        if args.method == "nagphormer":
            run(
                args=[
                    "python",
                    "src/NAGphormer/train.py",
                    f"--dataset={args.dataset}",
                    f"--path={args.path}",
                    f"--log_path={args.log_path}",
                    f"--device={args.device}",
                    f"--split=1.0",
                    "--patience=30",
                    "--batch_size=2000",
                    "--dropout=0.1",
                    "--hidden_dim=512",
                    "--hops=3",
                    "--n_heads=8",
                    f"--n_layers={n_layers}",
                    "--pe_dim=3",
                    "--peak_lr=0.01",
                    "--weight_decay=1e-05",
                ]
            )

        elif args.method == "graphsage":
            run(
                args=[
                    "python",
                    "src/graphsage/train.py",
                    f"--dataset={args.dataset}",
                    f"--path={args.path}",
                    f"--log_path={args.log_path}",
                    f"--device={args.device}",
                    f"--split=1.0",
                    "--dropout=0.1",
                    "--hidden_dim=64",
                    f"--n_layers={n_layers}",
                    "--lr=1e-3",
                    "--weight_decay=1e-05",
                ]
            )


# * >>> Plots
if args.experiment == "default":
    run(
        args=[
            "python",
            "src/plot-default.py",
            f"-i={args.log_path}/",
            f"-o={args.plot_path}/"
        ]
    )

else:
    run(
        args=[
            "python",
            "src/plot.py",
            f"-i={args.log_path}/",
            f"-o={args.plot_path}/",
            f"-e={args.experiment}"
        ]
    )
