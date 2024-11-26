# Implementation of CenNorm

## Requirements:
torch_geometric==2.5.3

numpy==1.24.3

scipy==1.10.1

torch-sparse==0.6.17+pt20cu118

torch==2.0.1

## Demo
python CenNorm.py --dataset yelp2018 --au 1 

python CenNorm.py --dataset amazon-book --au 6

python CenNorm.py --dataset iFashion --au 0.8

## Baselines
We offer our re-implementation of LightGCN SGL RGCF SimGCL in this repo.

For other baselines or origin implmention/re-implementation of them:

XSimGCL/SimGCL: [https://github.com/Coder-Yu/SELFRec](https://github.com/Coder-Yu/SELFRec)

SGL: [https://github.com/wujcan/SGL-Torch](https://github.com/wujcan/SGL-Torch)

LightGCN: [https://github.com/gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)

RGCF: [https://github.com/ChangxinTian/RGCF](https://github.com/ChangxinTian/RGCF)

DirectAU: [https://github.com/THUwangcy/DirectAU](https://github.com/THUwangcy/DirectAU)

GTN: [https://github.com/wenqifan03/GTN-SIGIR2022](https://github.com/wenqifan03/GTN-SIGIR2022)

CaGCN: [https://github.com/YuWVandy/CAGCN](https://github.com/YuWVandy/CAGCN)

SimpleX: [https://github.com/reczoo/RecZoo](https://github.com/reczoo/RecZoo)

