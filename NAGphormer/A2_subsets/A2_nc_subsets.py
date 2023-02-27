
import torch
import torch
from torch.nn import Linear
import torch.nn.functional as F
from data_utils_nc import load_data
import argparse
import math
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="name of the dataset",
                    type=str)

args = parser.parse_args()
dataset =load_data(args.dataset)
print()
print(f'Dataset: {dataset}:')
print('======================')
# print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print('dataset' , dataset)

if args.dataset =='Wisconsin':
    print('changed dim')
    data.train_mask = data.train_mask[:,0]#
    data.val_mask = data.val_mask[:,0]#
    data.test_mask = data.test_mask[:,0]#

print(data)
print('===========================================================================================================')

# Stats about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print("train dataset, val dataset and test dataset ", data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())

train_indices = (data.train_mask==True).nonzero().flatten()
# print('data.train_mask ', data.train_mask.sum())
print('train_indices ', train_indices)
unique_y = torch.unique(data.y)
print('unique_y', unique_y)
print('datay', data.y, data.y[train_indices])


# you can either use fraction or a fixed number.
fraction_remove=0.2 # this percentage will be removed in each iteration from the new train set. so first time 20%, next time 20% of the remaining 80%. Its upto you how you want to do this.

for i in range(0,2):
    print('\n\n removal iteration', i)
    updated_train_indices = []
    data.train_mask =torch.Tensor([False]*len(data.train_mask))

    # remove same number/fraction of elements from each class
    for cur_y in unique_y:
        print('\nclass is ', cur_y)
        # print('train_indices', len(train_indices))
        indices_cur_y =   (data.y==cur_y).nonzero().flatten()
        train_indices_y = np.intersect1d(indices_cur_y, train_indices)
        # print('train_indices_y', train_indices_y , len(train_indices_y))
        print('train_indices_y' , len(train_indices_y))

        # you can either use fraction or fixed number at each time. its your choice.
        num_elem_from_y_to_remove = math.ceil(fraction_remove*len(train_indices_y))
        print('num_elem_from_y_to_remove ', num_elem_from_y_to_remove)
        # elem_to_remove = train_indices_y[-num_elem_from_y_to_remove:]
        # print(' num elem_to_remove', len(elem_to_remove))
        
        #removed last 'num_elem_from_y_to_remove' elements from training for this class(cur_y)
        train_indices_y_updated = train_indices_y[0:len(train_indices_y) - num_elem_from_y_to_remove]
        print('train_indices_y_updated len ', len(train_indices_y_updated))

        print('train_indices len ', len(train_indices))
        updated_train_indices.extend(train_indices_y_updated)
    
    # print('updated_train_indices', updated_train_indices)
    print(' len updated_train_indices', len(updated_train_indices))

    #train indices after first step of removal
    train_indices = torch.Tensor(updated_train_indices).to(torch.int64)
    data.train_mask[train_indices]= True # updated train mask
    
    print(" updated train dataset, val dataset and test dataset ", data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())
        
