# Adversarially Regularized Graph Attention Networks for Inductive Learning on Partially Labeled Graphs

This is our implementation for the following paper:

>[Xiao, Jiaren, Quanyu Dai, Xiaochen Xie, James Lam, and Ka-Wai Kwok. "Adversarially regularized graph attention networks for inductive learning on partially labeled graphs." arXiv preprint arXiv:2106.03393 (2021)](https://arxiv.org/abs/2106.03393).


## Abstract
This paper studies the problem of cross-network node classification to overcome the insufficiency of labeled data in a single network. It aims to leverage the label information in a partially labeled source network to assist node classification in a completely unlabeled or partially labeled target network. Existing methods for single network learning cannot solve this problem due to the domain shift across networks. Some multi-network learning methods heavily rely on the existence of cross-network connections, thus are inapplicable for this problem. To tackle this problem, we propose a novel graph transfer learning framework AdaGCN by leveraging the techniques of adversarial domain adaptation and graph convolution. It consists of two components: a semi-supervised learning component and an adversarial domain adaptation component. The former aims to learn class discriminative node representations with given label information of the source and target networks, while the latter contributes to mitigating the distribution divergence between the source and target domains to facilitate knowledge transfer. Extensive empirical evaluations on real-world datasets show that AdaGCN can successfully transfer class information with a low label rate on the source network and a substantial divergence between the source and target domains.

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
