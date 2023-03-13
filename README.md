# Adversarially Regularized Graph Attention Networks for Inductive Learning on Partially Labeled Graphs

This is our implementation for the following paper:

>[Jiaren Xiao, Quanyu Dai, Xiaochen Xie, James Lam, and Ka-Wai Kwok. "Adversarially regularized graph attention networks for inductive learning on partially labeled graphs." Knowledge-Based Systems (2023)](https://www.sciencedirect.com/science/article/pii/S095070512300206X).


## Abstract
The high cost of data labeling often results in node label shortage in real applications. To improve node classification accuracy, graph-based semi-supervised learning leverages the ample unlabeled nodes to train together with the scarce available labeled nodes. However, most existing methods require the information of all nodes, including those to be predicted, during model training, which is not practical for dynamic graphs with newly added nodes. To address this issue, an adversarially regularized graph attention model is proposed to classify newly added nodes in a partially labeled graph. An attention-based aggregator is designed to generate the  representation of a node by aggregating information from its neighboring nodes, thus naturally generalizing to previously unseen nodes. In addition, adversarial training is employed to improve the model's robustness and generalization ability by enforcing node representations to match a prior distribution. Experiments on real-world datasets demonstrate the effectiveness of the proposed method in comparison with the state-of-the-art methods.

## Environment requirement
The codes can be run with the below packages:
* python == 3.7.9
* torch == 1.1.0 
* numpy == 1.15.4
* networkx == 1.9.1
* scipy == 1.5.4


## Datasets
Hyperlink: https://pan.baidu.com/s/1u198vHmQyI4DZI9tJ2jwUA

Password: 7p3c

## Examples to run the codes
* Performance Study (CiteSeer)
```
python main_AGAIN.py --label_per_class 20 --niters_gan_d 1 --lr_gan_d 0.001 --weight_decay 5e-2
```

## Citation 
If you would like to use our code, please cite:
```
@article{XIAO2023110456,
title = {Adversarially regularized graph attention networks for inductive learning on partially labeled graphs},
journal = {Knowledge-Based Systems},
year = {2023},
doi = {https://doi.org/10.1016/j.knosys.2023.110456},
author = {Xiao, Jiaren and Dai, Quanyu and Xie, Xiaochen and Lam, James and Kwok, Ka-Wai},
}
```
