# Adversarially Regularized Graph Attention Networks for Inductive Learning on Partially Labeled Graphs

This is our implementation for the following paper:

>[Xiao, Jiaren, Quanyu Dai, Xiaochen Xie, James Lam, and Ka-Wai Kwok. "Adversarially regularized graph attention networks for inductive learning on partially labeled graphs." arXiv preprint arXiv:2106.03393 (2021)](https://arxiv.org/abs/2106.03393).


## Abstract
Graph embedding is a general approach to tackling graph-analytic problems by encoding nodes into low-dimensional representations. Most existing embedding methods are transductive since the information of all nodes is required in training, including those to be predicted. In this paper, we propose a novel inductive embedding method for semi-supervised learning on graphs. This method generates node representations by learning a parametric function to aggregate information from the neighborhood using an attention mechanism, and hence naturally generalizes to previously unseen nodes. Furthermore, adversarial training serves as an external regularization enforcing the learned representations to match a prior distribution for improving robustness and generalization ability. Experiments on real-world clean or noisy graphs are used to demonstrate the effectiveness of this approach. 

## Environment requirement
The code has been tested running under Python 3.5.2. The required packages are as follows:
* python == 3.5.2
* tensorflow-gpu == 1.13.0-rc0 
* numpy == 1.16.2

## Examples to run the codes
* Multi-label classification with source training rate as 10% (Table 3)
```
python train_WD.py # set signal = [1], target_train_rate = [0], FLAGS.gnn=gcn or FLAGS.gnn=igcn
```

* Multi-label classification with source training rate as 10% and target train set as 5% (Table 4)
```
python train_WD.py # set signal = [2], target_train_rate = [0.05], FLAGS.gnn=gcn or FLAGS.gnn=igcn
```

## Citation 
If you would like to use our code, please cite:
```
@article{dai_graph_2022,
	title = {Graph transfer learning via adversarial domain adaptation with graph convolution},
	author = {Dai, Quanyu and Wu, Xiao-Ming and Xiao, Jiaren and Shen, Xiao and Wang, Dan},
	year = {2022},
	pages = {1--1},
}
```
