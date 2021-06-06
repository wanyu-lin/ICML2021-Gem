# Gem
This repo covers the implementation for our paper Gem.

Wanyu Lin, Hao Lan, and Baochun Li. "Generative Causal Explanations for Graph Neural Networks (https://arxiv.org/pdf/2104.06643.pdf)," in the Proceedings of the 38th International Conference on Machine Learning (ICML 2021), Online, July 18-24, 2021.


## Download code and apply patch for GNNExplainer
```sh
git clone https://github.com/wanyu-lin/ICML2021-Gem Gem
cd Gem
git submodule init
cd gnnexp
git apply ../gnnexp.patch
```

## Setup environment
Create an environment with conda:
```sh
conda create -n gem python=3.8.8
conda activate gem
```
Install PyTorch with CUDA 10.2:
```sh
conda install pytorch cudatoolkit=10.2 -c pytorch
```
Or install PyTorch WITHOUT CUDA:
```sh
conda install pytorch cpuonly -c pytorch
```
Install other required packages:
```sh
conda install opencv scikit-learn networkx pandas matplotlib seaborn
pip install tensorboardx
```

## Train classification models with GNNExplainer's code

```sh
cd gnnexp
python gnnexp/train.py --dataset=syn1
python gnnexp/train.py --dataset=syn4
python gnnexp/train.py --bmname=Mutagenicity
python gnnexp/train.py --bmname=NCI1
```
or you can directly use the checkpoint
```sh
unzip ckpt.zip
```

## Generate explaination of classification models by GNNExplainer
```sh
python gnnexp/explainer_main.py --dataset=syn1 --logdir=explanation/gnnexp
python gnnexp/explainer_main.py --dataset=syn4 --logdir=explanation/gnnexp
python gnnexp/explainer_main.py --dataset=Mutagenicity --graph-mode --logdir=explanation/gnnexp
python gnnexp/explainer_main.py --dataset=NCI1 --graph-mode --logdir=explanation/gnnexp
```
or you can directly unzip the explanation by
```sh
unzip gnnexp_explanation.zip
```

## Distillation
```sh
python generate_ground_truth.py --dataset=syn1 --top_k=6
python generate_ground_truth.py --dataset=syn4 --top_k=6
python generate_ground_truth_graph_classification.py --dataset=Mutagenicity --output=mutag --graph-mode --top_k=20
python generate_ground_truth_graph_classification.py --dataset=NCI1 --output=nci1_dc --graph-mode --top_k=20 --disconnected
```
or you can directly extract from zip file
```
unzip distillation.zip
```

## Train Gem
```sh
python explainer_gae.py --dataset=syn1 --distillation=syn1_top6 --output=syn1_top6
python explainer_gae.py --dataset=syn4 --distillation=syn4_top6 --output=syn4_top6
python explainer_gae_graph.py --distillation=mutag_top20 --output=mutag_top20 --dataset=Mutagenicity --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01
python explainer_gae_graph.py --distillation=nci1_dc_top20 --output=nci1_dc_top20 --dataset=NCI1 --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01
```

## Evaluate GNNExplainer and Gem
```sh
python test_explained_adj.py --dataset=syn1 --distillation=syn1_top6 --exp_out=syn1_top6 --top_k=6
python test_explained_adj.py --dataset=syn4 --distillation=syn4_top6 --exp_out=syn4_top6 --top_k=6
python test_explained_adj_graph.py --graph-mode --dataset=Mutagenicity --exp_out=mutag_top20 --distillation=mutag_top20 --top_k=15 --test_out=mutag_top20_top15
python test_explained_adj_graph.py --graph-mode --dataset=NCI1 --exp_out=nci1_dc_top20 --distillation=nci1_dc_top20 --top_k=15 --test_out=nci1_dc_top20_top15
```


## Visualization
Run `*.ipynb` files in Jupyter Notebook or Jupyter Lab.


## Reference
If you make advantage of Gem in your research, please cite the following in your manuscript:

```
@inproceedings{
wanyu-icml21,
title="{Generative Causal Explanations for Graph Neural Networks}",
author={Lin, Wanyu and Lan, Hao and Li, Baochun},
booktitle={International Conference on Machine Learning },
year={2021},
url={https://arxiv.org/pdf/2104.06643.pdf},
}
```
