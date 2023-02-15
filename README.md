# Adversarially Regularized Graph Attention Networks for Inductive Learning on Partially Labeled Graphs

This is our implementation for the following paper:

>[Jiaren Xiao, Quanyu Dai, Xiaochen Xie, James Lam, and Ka-Wai Kwok. "Adversarially regularized graph attention networks for inductive learning on partially labeled graphs." arXiv preprint arXiv:2106.03393 (2021)](https://arxiv.org/abs/2106.03393).


## Abstract
Graph embedding is a general approach to tackling graph-analytic problems by encoding nodes into low-dimensional representations. Most existing embedding methods are transductive since the information of all nodes is required in training, including those to be predicted. In this paper, we propose a novel inductive embedding method for semi-supervised learning on graphs. This method generates node representations by learning a parametric function to aggregate information from the neighborhood using an attention mechanism, and hence naturally generalizes to previously unseen nodes. Furthermore, adversarial training serves as an external regularization enforcing the learned representations to match a prior distribution for improving robustness and generalization ability. Experiments on real-world clean or noisy graphs are used to demonstrate the effectiveness of this approach. 

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
@article{xiao_adversarially_2021,
	title = {Adversarially regularized graph attention networks for inductive learning on partially labeled graphs},
	journal = {arXiv:2106.03393 [cs]},
	author = {Xiao, Jiaren and Dai, Quanyu and Xie, Xiaochen and Lam, James and Kwok, Ka-Wai},
	year = {2021},
}
```
