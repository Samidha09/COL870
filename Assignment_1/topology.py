import torch
from torch.nn import Linear
import torch.nn.functional as F
from data_utils import load_data
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from model import *
import argparse
import copy
import tqdm
import os

isExist = os.path.exists('./topology_generated')
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs('topology_generated')
   print("The new directory is created!")

#exp1 - only mlp vs gnn

#exp2 - vary number of gnn layers