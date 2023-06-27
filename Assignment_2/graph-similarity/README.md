## GRAPH SIMILARITY LEARNING

Datasets: \
LINUX \
AIDS700nef 

We have benchmarked two architectures, namely GraphSAGE and NAGphormer for the graph similarity learning task.

### GraphSAGE
**Commands to run the code:**

``` bash
cd MPNN 

python src/main.py --dataset <dataset-name> --gnn-operator sage --plot --save best_model_<dataset-name>.pt --epochs 500
```

Sources: 
https://github.com/gospodima/Extended-SimGNN/tree/master/src


### NAGphormer

Recommended directory structure:
- `cd NAGphormer`
- Run `python src/create-dirs.py -f ./` to create the recommended directory structure.
- It creates two folders (and their subfolders): `log`, `plots`.
- These provide convenient options for `--path, --log_path, --plot_path` in the experiment's arguments.

``` bash
cd NAGphormer 

python src/NAGphormer/train.py --dataset <dataset-name> --path data/nagphormer/ --pyg_path data/pyg/<dataset-name>/ --dropout 0.1 --hidden_dim 512 --hops 3 --n_head 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01 --weight_decay 1e-05 --device 1 --log_path log/nagphormer/<dataset-name> --epochs 1000
```
Please run `python src/NAGphormer/train.py -h` to see the full list of args. 

Source: https://github.com/JHL-HUST/NAGphormer

> **_NOTE:_**  Plots will be saved in **plots** folder inside respective architecture's directory
