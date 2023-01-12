from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import os

def load_data(dataset_name):
    if dataset_name == 'Cora':
        path = os.path.join(os.getcwd(), 'data','Cora')
        dataset = Planetoid(path,name='Cora')
        return dataset
    
    if dataset_name == 'CiteSeer':
        path = os.path.join(os.getcwd(), 'data','CiteSeer')
        dataset = Planetoid(path,name='CiteSeer', transform=NormalizeFeatures())
        return dataset
    
    