"""Run experiments to generate log files."""
from argparse import ArgumentParser
from subprocess import call


# * >>> Arguments.
parser = ArgumentParser()
parser.add_argument("--experiment", "-e", dest="experiment", type=str,
                    choices=[
                        "default", "one_hot",
                        "training-data", "hops", "pe_dim",
                        "hidden_dim", "n_layers", "top_k",
                    ],
                    help="Which experiment to call.")
parser.add_argument("--dataset", "-d", dest="dataset", type=str,
                    choices=["LINUX", "AIDS700nef"],
                    help="Name of the dataset.")
parser.add_argument("--path", "-p", type=str,
                    help="Path to folder containg the dataset."\
                        "If 'LINUX.pt' is in 'folder/', use '-p=folder/'.")
parser.add_argument("--pyg_path", type=str,
                    help="Path to folder containg the pyg dataset."\
                        "If 'LINUX/' is in 'folder/', use '-p=folder/LINUX'.")
parser.add_argument("--log_path", "-lp", type=str, help="Where to store log.csv.")
parser.add_argument("--plot_path", "-pp", type=str, help="Where to save the plots.")
parser.add_argument("--device", type=int, help="Device cuda ID.")

args = parser.parse_args()
print("\n>>> run.py:", args)

training_data_splits = [1.]
if args.experiment == "training-data":
    training_data_splits = [0.2, 0.4, 0.6, 0.8, 1.0]


# * >>> Benchmarking.
if args.experiment == "default" or args.experiment == "training-data":
    for t_split in training_data_splits:
        print(f"\n>>> Training split = {t_split}")
        call(
            args=[
                "python",
                "src/NAGphormer/train.py",
                f"--dataset={args.dataset}",
                f"--path={args.path}",
                f"--log_path={args.log_path}",
                f"--device={args.device}",
                f"--split={t_split}",
                "--patience=30",
                "--batch_size=32",
                "--dropout=0.1",
                "--hidden_dim=128",
                "--hops=3",
                "--n_heads=8",
                "--n_layers=1",
                "--pe_dim=3",
                "--peak_lr=0.001",
                "--weight_decay=1e-05",
                f"--pyg_path={args.pyg_path}",
                "--top_k=10",
            ]
        )

elif args.experiment == "hops":
    for hops in range(1, 4):
        call(
            args=[
                "python",
                "src/NAGphormer/train.py",
                f"--dataset={args.dataset}",
                f"--path={args.path}",
                f"--log_path={args.log_path}",
                f"--device={args.device}",
                f"--split=1.0",
                "--patience=30",
                "--batch_size=32",
                "--dropout=0.1",
                "--hidden_dim=128",
                f"--hops={hops}",
                "--n_heads=8",
                "--n_layers=1",
                "--pe_dim=3",
                "--peak_lr=0.001",
                "--weight_decay=1e-05",
                f"--pyg_path={args.pyg_path}",
                "--top_k=10",
            ]
        )

elif args.experiment == "pe_dim":
    for pe_dim in [1, 2, 3, 4]:
        call(
            args=[
                "python",
                "src/NAGphormer/train.py",
                f"--dataset={args.dataset}",
                f"--path={args.path}",
                f"--log_path={args.log_path}",
                f"--device={args.device}",
                f"--split=1.0",
                "--patience=30",
                "--batch_size=32",
                "--dropout=0.1",
                "--hidden_dim=128",
                f"--hops=3",
                "--n_heads=8",
                "--n_layers=1",
                f"--pe_dim={pe_dim}",
                "--peak_lr=0.001",
                "--weight_decay=1e-05",
                f"--pyg_path={args.pyg_path}",
                "--top_k=10",
            ]
        )

elif args.experiment == "hidden_dim":
    for hidden_dim in [32, 64, 128, 256, 512]:
        call(
            args=[
                "python",
                "src/NAGphormer/train.py",
                f"--dataset={args.dataset}",
                f"--path={args.path}",
                f"--log_path={args.log_path}",
                f"--device={args.device}",
                f"--split=1.0",
                "--patience=30",
                "--batch_size=32",
                "--dropout=0.1",
                f"--hidden_dim={hidden_dim}",
                "--hops=3",
                "--n_heads=8",
                "--n_layers=1",
                "--pe_dim=3",
                "--peak_lr=0.001",
                "--weight_decay=1e-05",
                f"--pyg_path={args.pyg_path}",
                "--top_k=10",
            ]
        )

elif args.experiment == "n_layers":
    for n_layers in range(1, 6):
        call(
            args=[
                "python",
                "src/NAGphormer/train.py",
                f"--dataset={args.dataset}",
                f"--path={args.path}",
                f"--log_path={args.log_path}",
                f"--device={args.device}",
                f"--split=1.0",
                "--patience=30",
                "--batch_size=32",
                "--dropout=0.1",
                "--hidden_dim=128",
                "--hops=3",
                "--n_heads=8",
                f"--n_layers={n_layers}",
                "--pe_dim=3",
                "--peak_lr=0.001",
                "--weight_decay=1e-05",
                f"--pyg_path={args.pyg_path}",
                "--top_k=10",
            ]
        )

elif args.experiment == "top_k":
    for top_k in [1, 5, 15, 20]:
        call(
            args=[
                "python",
                "src/NAGphormer/train.py",
                f"--dataset={args.dataset}",
                f"--path={args.path}",
                f"--log_path={args.log_path}",
                f"--device={args.device}",
                f"--split=1.0",
                "--patience=30",
                "--batch_size=32",
                "--dropout=0.1",
                "--hidden_dim=128",
                "--hops=3",
                "--n_heads=8",
                f"--n_layers=1",
                "--pe_dim=3",
                "--peak_lr=0.001",
                "--weight_decay=1e-05",
                f"--pyg_path={args.pyg_path}",
                f"--top_k={top_k}",
            ]
        )

elif args.experiment == "one_hot":
    call(
        args=[
            "python",
            "src/NAGphormer/train.py",
            f"--dataset={args.dataset}",
            f"--path={args.path}",
            f"--log_path={args.log_path}",
            f"--device={args.device}",
            f"--split=1.0",
            "--patience=30",
            "--batch_size=32",
            "--dropout=0.1",
            "--hidden_dim=128",
            "--hops=3",
            "--n_heads=8",
            "--n_layers=1",
            "--pe_dim=3",
            "--peak_lr=0.001",
            "--weight_decay=1e-05",
            f"--pyg_path={args.pyg_path}",
            "--top_k=10",
            "--one_hot=True"
        ]
    )


# * >>> Plots
if args.experiment in ["default", "one_hot"]:
    call(
        args=[
            "python",
            "src/plot-default.py",
            f"-i={args.log_path}/",
            f"-o={args.plot_path}/"
        ]
    )

else:
    call(
        args=[
            "python",
            "src/plot.py",
            f"-i={args.log_path}/",
            f"-o={args.plot_path}/",
            f"-e={args.experiment}"
        ]
    )
