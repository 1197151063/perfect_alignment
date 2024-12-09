import world
from model import LightGCN
from dataloader import Loader
import torch
from procedure import train_bpr,test
import utils
from torch_sparse import matmul
from torch_geometric.utils import degree
from tqdm import tqdm
device = world.device
config = world.config
dataset = Loader()
num_users = dataset.num_users
num_items = dataset.num_items
utils.set_seed(config['seed'])
t_edge_index = dataset.train_edge_index
# user_degree = degree(t_edge_index[0])
# user_mask = user_degree <= 20
# user_idx = torch.arange(0,num_users)
# item_degree = degree(t_edge_index[1])
# user_idx = user_idx[user_mask]
# print(user_idx)
# print(user_mask)
# adj_mat = torch.sparse_coo_tensor(t_edge_index,torch.ones(t_edge_index[0].size(0)),(num_users,num_items)).to_dense().T
# # tail_adj_mat = adj_mat[user_idx,:]
# print(adj_mat.shape)
# user_user_common = adj_mat @ adj_mat.T
# print(user_user_common.shape)
# # user_user_common = user_user_common[user_idx,:]
# user_item_counts = adj_mat.sum(dim=1)
# print(user_item_counts.shape)
# all_neighbors_matrix = user_item_counts.view(-1, 1) + user_item_counts.view(1, -1) - user_user_common
# u_u_co = user_user_common / all_neighbors_matrix.float()
# print(u_u_co.shape)
# value, knn_ind = torch.topk(u_u_co, 11, dim=-1)
# torch.save(value,'i-i-value.pt')
# torch.save(knn_ind,'i-i-index.pt')
# print(value,knn_ind)
value = torch.load('i-i-value.pt')
knn_index = torch.load('i-i-index.pt')

value = value[:,1:].to(device)
value[value < 0.1] = 0
knn_index = knn_index[:,1:].to(device)
print(value.shape)
print(knn_index.shape)
# scr_nodes = knn_ind[:, 0]  
# print(scr_nodes)
# dst_nodes = knn_ind[:, 1:]  
# val = value[:,1:]
# row_indices = scr_nodes.unsqueeze(1).repeat(1, dst_nodes.size(1)).flatten()
# print(row_indices)
# col_indices = dst_nodes.flatten()
# val = val.flatten()
# masker = val >= 0.1
# r = row_indices[masker].long()
# c = col_indices[masker].long()
# v = val[masker]
# h_index = torch.stack([r,c]).to(device)
# v = v.to(device)
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)

model = LightGCN(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config,
                 knn_ind=knn_index,
                 value=value).to(device)
opt = torch.optim.Adam(params=model.parameters(),lr=config['lr'])
best = 0.
patience = 0.
max_score = 0.
for epoch in range(1, 1001):
    loss = train_bpr(dataset=dataset,model=model,opt=opt)
    recall,ndcg = test([20],model,train_edge_index,test_edge_index,num_users)
    flag,best,patience = utils.early_stopping(recall[20],ndcg[20],best,patience,model)
    if flag == 1:
        break
    print(f'Epoch: {epoch:03d}, {loss}, R@20: '
          f'{recall[20]:.4f},  '
          f'N@20: {ndcg[20]:.4f}')