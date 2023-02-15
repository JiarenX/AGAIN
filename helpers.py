"""
    helpers.py
"""

from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable


def set_seeds(seed=0):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)


def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


def label_node(nodes, targets, ins_num):
    train_tar = targets[nodes]
    n_classes = list(set(targets[:, -1]))
    class_idx = {}
    train_nodes = []
    for i in n_classes:
        class_idx[i] = np.where(np.array(train_tar) == i)[0][:ins_num]
        train_nodes.append(nodes[class_idx[i]])

    return np.hstack(train_nodes)