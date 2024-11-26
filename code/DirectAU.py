from world import cprint,bprint
import world
from re_model import AlignGCN
from re_dataloader import Loader
import torch
import numpy as np
from re_procedure import train_bpr,test,Test,train_bpr_aligngcn
import re_procedure
import random 
import utils
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import sys
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# file_name = f"result_{current_time}.txt"
# file = open(file_name, "w")
# sys.stdout = file
device = world.device
config = world.config
dataset = Loader()
utils.set_seed(config['seed'])
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = AlignGCN(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.

for epoch in range(1, 1001):
    loss = train_bpr_aligngcn(dataset=dataset,model=model,opt=opt)
    recall,ndcg = test([20],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        # file.close()
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f},  '
          f'N@20: {ndcg[20]:.4f}')