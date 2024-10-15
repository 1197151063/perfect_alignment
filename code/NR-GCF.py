from world import cprint,bprint
import world
from re_model import NR_GCF,LightGCN
from re_dataloader import Loader
import torch
import numpy as np
from re_procedure import train_bpr,test,Test,train_bpr_nrgcf
import re_procedure
import random 
import utils
import numpy as np

device = world.device
config = world.config
dataset = Loader()
utils.set_seed(config['seed'])
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model0 = LightGCN(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt0 = torch.optim.Adam(params=model0.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
mem_loss = torch.zeros_like(train_edge_index[0]).to(device)
t = 10
"""
First Denoising Stage
"""
for epoch in range(1, t):
    model0.train()
    neg_ind = torch.randint(0,num_items,(train_edge_index[0].numel(),)).unsqueeze(0).to(device)
    ins_index = torch.cat([train_edge_index,neg_ind],dim=0)
    instance_loss = model0.instance_loss(ins_index)
    mem_loss = (epoch / 10) * mem_loss + (1 - epoch/10) * instance_loss
    S = utils.Fast_Sampling(dataset=dataset)
    aver_loss = 0.
    total_batch = len(S)
    for edge_label_index in S:
        pos_rank,neg_rank = model0(edge_label_index)
        bpr_loss,reg_loss = model0.recommendation_loss(pos_rank,neg_rank,edge_label_index)
        loss = bpr_loss + reg_loss
        opt0.zero_grad()
        loss.backward()
        opt0.step()
        aver_loss += loss.cpu().item()
    aver_loss /= total_batch
    recall,ndcg = test([20,50],model0,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model0)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, loss: {loss:5f}, R@20: '
          f'{recall[20]:.4f}, R@50: {recall[50]:.4f} '
          f', N@20: {ndcg[20]:.4f}, N@50: {ndcg[50]:.4f}')

cprint("First stage ended")
"""
Graph Denoising
"""
threshold = (config['beta'] + 1) * mem_loss.median()
mem_mask = (mem_loss <= threshold)
print(train_edge_index.size(1))
train_edge_index = train_edge_index[:,mem_mask]
print(train_edge_index.size(1))
"""
secone stage training
"""
model = NR_GCF(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])

for epoch in range(t,1000):
    loss = train_bpr_nrgcf(dataset=dataset,model=model,opt=opt)
    recall,ndcg = test([20,50],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f}, R@50: {recall[50]:.4f} '
          f', N@20: {ndcg[20]:.4f}, N@50: {ndcg[50]:.4f}')
