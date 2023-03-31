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
``` bash
cd NAGphormer 
```

Source: https://github.com/JHL-HUST/NAGphormer

> **_NOTE:_**  Plots will be saved in **plots** folder inside respective architecture's directory