import common 
from model import LightGCN
import numpy as np
import torch
import utils
from dataloader import Loader
from procedure import test,trainer
from common import cprint

args = common.args
device = common.device
seed = common.seed

utils.seed_everything(seed)

dataset = Loader()

num_users = dataset.num_users
num_items = dataset.num_items
edge_index = dataset.adj_mat.to(device)
adj_mat = dataset.bipartite_graph.to(device)

rec_model = LightGCN(num_users=num_users,
                     num_items=num_items,
                     edge_index=edge_index,
                     adj_mat=adj_mat)
rec_model = rec_model.to(device)
loss = utils.BPR(rec_model)
best = 0
score = 0
for epoch in range(common.TRAIN_epochs):
    loss_info = trainer(dataset,rec_model,loss)
    cprint("[TEST]")
    pre,recall,ndcg = test(rec_model,dataset)
    score = recall + ndcg 
    if score > best:
        best = score
        cprint("best epoch for now")
    topk_txt = f'Testing EPOCH[{epoch + 1}/{common.TRAIN_epochs}] {loss_info} | Results Top-k (pre, recall, ndcg): {pre}, {recall}, {ndcg}'
    print(topk_txt)