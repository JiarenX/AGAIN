# GAT Codes for Inductive Learning

This is an implementation of the GAT model (https://arxiv.org/abs/1710.10903) for inductive node classfication on a single graph.

The repo has been forked initially from https://github.com/Diego999/pyGAT. The official repository for the GAT (Tensorflow) is available in https://github.com/PetarV-/GAT.

# Requirements

The codes can be run with the below packages:
* python == 3.7.9
* torch == 1.7.1+cu101 
* numpy == 1.15.4
* networkx == 1.9.1
* scipy == 1.5.4

# Example
* Performance Study (CiteSeer)
```
python train.py
```

Note: The datasets here are exactly the same as the ones in our AGAIN codes, although the formats are different.
