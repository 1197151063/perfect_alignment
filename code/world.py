import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing
from torch_sparse import SparseTensor
import random
import numpy as np
args = parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys

sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['gowalla', 'yelp2018', 'amazon-book', 'last-fm']
all_models = ['mf', 'gtn', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['K'] = args.K
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['args'] = args
config['dataset'] = args.dataset
config['epochs'] = args.epochs
config['lambda2'] = args.lambda2
config['add_noise'] = args.add_noise
config['noise_rate'] = args.noise_rate
config['lightGCN_n_layers']= args.layer
config['model']=args.model 
config['seed']=args.seed
config['norm_weight']=args.norm_weight
config['normal_weight']=args.normal_weight
config['lambda1'] = 32
config['beta'] = args.pruning
config['ssl_tmp'] = args.ssl_tmp
config['ssl_decay'] = args.ssl_decay
config['epsilon'] = args.epsilon
GPU = torch.cuda.is_available()
torch.cuda.set_device(args.gpu_id)
device = torch.device('cuda' if GPU else "cpu")
# device = torch.device("cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = [20,50]
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

flag = 0
def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

def bprint(words:str):
    print(f"\033[0;30;45m{words}\033[0m")