from world import cprint,bprint
import world
from re_model import SGL
from re_dataloader import Loader
import torch
import numpy as np
from re_procedure import train_bpr_sgl,test,Test
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import re_procedure
import random 
import utils
import numpy as np
from torch_geometric.utils import dropout_edge
utils.set_seed(42)
device = world.device
config = world.config
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = SGL(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.

for epoch in range(1, 1001):
    edge_index = train_edge_index
    edge_index1,_ = dropout_edge(edge_index=edge_index,p=0.1)
    edge_index2,_ = dropout_edge(edge_index=edge_index,p=0.1)
    edge_index1 = model.get_sparse_graph(edge_index1)
    edge_index2 = model.get_sparse_graph(edge_index2)
    edge_index1 = gcn_norm(edge_index1)
    edge_index2 = gcn_norm(edge_index2)
    loss = train_bpr_sgl(dataset=dataset,
                         model=model,
                         opt=opt,
                         edge_index1=edge_index1,
                         edge_index2=edge_index2)
    recall,ndcg = test([20],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f} '
          f', N@20: {ndcg[20]:.4f}')