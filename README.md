# NR-GCF Pytorch

## Official pytorch Implementation of NR-GCF

The original version of this code base was from LightGCN-pytorch: https://github.com/gusye1234/LightGCN-PyTorch.



## Abstract

Graph Neural Networks (GNNs) have emerged as the preferred backbone model of collaborative filtering, credited to their strong capability in capturing the intricate topological relationships in user-item interactions. Nevertheless, a common oversight in existing studies is the presumption of the inherent reliability of these interactions, ignoring the reality that a significant fraction of user-item engagements, such as accidental clicks, are inherently noisy. Extensive studies have revealed that, GNN structure is vulnerable to such noisy edges within the graph-structured data, as those noisy edges can mislead the network into overfitting incorrect patterns of interactions, thereby propagating such incorrect information through entire interaction network. To address those challenges, in this paper, we propose a novel noise-robust GNN based recommender system, known as Noise-resistant Graph Collaborative Filtering (NR-GCF). NR-GCF innovatively employs an adaptive filtering threshold to sieve reliable edges based on the memorization effect of GNN, and further utilizes the edge-dependent information to learn noise-resistant representation for robust recommendation purposes. Comprehensive experiments and ablation studies demonstrate the effectiveness and robustness of the proposed framework. Our implementation has been made available in the attachment.



## Requirements

torch_geometric==2.5.3

numpy==1.24.3

scipy==1.10.1

torch-sparse==0.6.17+pt20cu118

torch==2.0.1



## Run paper Statistics

### For clean datasets:

#### Yelp2018

```
python re_filter.py --dataset yelp2018 --pruning 0.2
```

#### Amazon-book

```
python re_filter.py --dataset amazon-book --lr 1e-4 --pruning 0.05
```

#### Gowalla

```
python re_filter.py --norm_weight 0 --normal_weight 1 --pruning 0.8 --dataset gowalla
```

### For 20% noise poisoned:

#### Yelp2018

```
python re_filter.py --add_noise 1 --noise_rate 0.2 --dataset yelp2018 --pruning 0.2
```

#### Amazon-book

```
python re_filter.py --add_noise 1 --noise_rate 0.2 --dataset amazon-book --pruning 0.2
```

#### Gowalla

```
python re_filter.py --lr 5e-4 --pruning 0.2 --add_noise 1 --noise_rate 0.2 --dataset gowalla
```

### For more noisy cases:

simply by using

```
--add_noise 1 --noise_rate 0.x
```



## Run full track of the paper

first run 

```
python pre_train.py 
```

then run 

```
python re_filter.py --dataset yelp2018
```

to run other cases

just select --dataset in [gowalla,yelp2018,amazon-book]



## Run baselines

Here we also offer our pytorch implementation of GTN, LightGCN, NGCF & MF

run them by using

```
python NGCF.py --decay 1e-5 
```

```
python GTN.py
```

```
python LightGCN.py
```

```
python BPRMF.py
```





