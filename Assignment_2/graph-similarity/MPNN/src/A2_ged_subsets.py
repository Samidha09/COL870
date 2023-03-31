r"""This script computes a subset of GED Dataset's training split whose
size is `ratio` times the full dataset size. The graph indices are stored
in an npy file.
"""
import argparse
import numpy as np
from data_utils_ged import load_ged_dataset


def get_subsets(name, ratio):
    r"""
    param `name`: name of GED Dataset. str. can be AIDS700nef or LINUX
    param `ratio`: float. can be in [0, 1].
    """
    if not (ratio >= 0 and ratio <= 1):
        raise ValueError(f"Ratio needs to be in [0, 1]. Found {ratio}.")
    dataset = load_ged_dataset(name)
    total = len(dataset)
    validation = {'AIDS700nef': 140, 'LINUX': 200}
    total = total - validation[name] # graphs from end will be in validation set
    perm = np.random.RandomState(seed=0).permutation(total)
    # A fixed seed=0 ensures larger sets are supersets of smaller sets
    subset_size = int(np.ceil(ratio * total))
    gids = perm[:subset_size]
    filename = f"./dataset/{name}/{name}_{ratio}_training_graph_idxs.npy"
    np.save(filename, gids)
    print(f"Graph indices saved in {filename}")


def main():
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser(
            description="This script computes a subset of GED Dataset's "
                        "training split whose size is `ratio` times the "
                        "full dataset size. The graph indices are stored "
                        "in an npy file."
            )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["AIDS700nef", "LINUX"],
                        help="the name of the GED Dataset"
                        )
    parser.add_argument("--ratio", type=float, required=True,
                        help="the amount of training dataset, expressed"
                        " as a ratio, to keep"
                        )
    pargs = parser.parse_args()
    # -------------------------------------------------------------
    get_subsets(pargs.dataset, pargs.ratio)


if __name__=="__main__":
    main()
