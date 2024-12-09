import world
from model import SGL
from dataloader import Loader
import torch
from procedure import train_bpr_sgl,test
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import utils
from torch_geometric.utils import dropout_edge
utils.set_seed(42)
device = world.device
config = world.config
dataset = Loader()
train_edge_index = dataset.train_edge_index.to(device)
test_edge_index = dataset.test_edge_index.to(device)
value = torch.load('i-i-value.pt')
knn_index = torch.load('i-i-index.pt')

value = value[:,1:].to(device)
value[value < 0.1] = 0
knn_index = knn_index[:,1:].to(device)
num_users = dataset.num_users
num_items = dataset.num_items
model = SGL(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config,
                 knn_ind=knn_index,
                 val=value).to(device)
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