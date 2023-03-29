# Benchmarking NAGphormer against GraphSAGE on node classification

<img src="https://github.com/Samidha09/COL870/blob/node-classification/Assignment_2/node-classification/f1-vs-epoch.png" width="90%"></img><img src="https://github.com/Samidha09/COL870/blob/node-classification/Assignment_2/node-classification/train-loss-vs-epoch.png" width="45%"></img><img src="https://github.com/Samidha09/COL870/blob/node-classification/Assignment_2/node-classification/val-loss-vs-epoch.png" width="45%"></img>

**NOTE:** All scripts must be run from `COL870/Assignment_2/node-classification/`.

# Environment
Run the following to create and activate an environment named `nag-node`:
- `conda env create -f env.yml`
- `conda activate nag-node`

# Recommended directory structure
## Data
- Run `python src/set-data-folder.py>`
- This will create a folder `data` in `node-classification/`.

## Logs and Plots
- Run `python src/create-dirs.py -f ./` to create the recommended directory structure.
- It creates two folders (and their subfolders): `log`, `plots`.
- These provide convenient options for `--path, --log_path, --plot_path` in the experiment's arguments.

# Custom directory structure
- Unzip data.zip whereever you wish and use that for `--path` when required..
- Supply custom paths to `--path, --log_path, --plot_path` when required.
- Or use `src/create-dirs.py` with custom value against the `-f` flag.

# Experiments
Following experiments are available for cora and wisconsin datasets:
1. `default` :
    - macro f1 vs epochs
    - macro f1 vs time
    - train/val loss vs time
2. `hidden_dim` : Varying the hidden dimension of the layers involved.
3. `hops` (nagphormer exclusive) : Varying the #hops used to compute the subgraphs in nagphormer.
4. `n_layers` : Varying the number of layers (transformer layers in nagphormer, SAGE layers in graphSAGE).
5. `pe_dim` (nagphormer exclusive): Varying the positional embedding dimension in nagphormer.
6. `training-data` : Varying the fraction of training data to be used for training.

# Run
- If one has followed the recommended directory structure and wants to run a specific experiment:
    - Run `python src/run-exp.py -h` to see the required arguments.
    - Run `src/run-exp.py` accordingly.
- If one has followed the recommended directory structure and wants to run all the experiments:
    - Run `python src/run-all.py`
- For fine tuned control use `src/run.py`
    - Please run `python src/run.py -h` to see the required arguments.
    - Note that all arguments are mandatory.
    - Run `src/run.py` accordingly.
