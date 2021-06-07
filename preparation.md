
## Train classification models

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

## Generate explaination of classification models with GNNExplainer
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