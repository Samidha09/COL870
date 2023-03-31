## GRAPH CLASSIFICATION TASK

Datasets: \
ogbg-ppa \
ogbg-molhiv 

We have benchmarked two architectures, namely GraphSAGE and NAGphormer for the graph classification task.

### GraphSAGE
**Commands to run the code:**

``` bash
cd MPNN 
python main_pyg.py --gnn sage --datsaset <dataset-name> 

```

Sources: 
https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/ppa

https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/sage_conv.html

### NAGphormer
``` bash
cd NAGphormer 
```
For ogbg-molhiv:
``` bash
python train.py --dataset ogbg-molhiv --batch_size 2000 --dropout 0.1 --hidden_dim 64 --hops 3  --n_heads 4 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --graph_batch_size 200 --num_classes 2
```
For ogbg-ppa:

``` bash
taskset -c 48-71 python train.py --dataset ogbg-ppa --batch_size 2000 --dropout 0.1 --hidden_dim 64 --hops 3  --n_heads 1 --n_layers 1 --pe_dim 10 --peak_lr 0.01  --weight_decay=1e-05 --graph_batch_size 1000 --num_classes 37

```
Source: https://github.com/JHL-HUST/NAGphormer
