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


train_edge_index = dataset.train_edge_index
test_edge_index = dataset.test_edge_index
num_users = dataset.num_users
num_items = dataset.num_items
model = SGL(num_users=num_users,
                 num_items=num_items,
                 edge_index=train_edge_index,
                 config=config)
model.load_state_dict(torch.load('./models/LightGCN-yelp2018.pth.tar',map_location=torch.device('cpu')))
users,items = torch.split(model.get_embedding(),[num_users,num_items])
pos_edges_index = torch.randperm(test_edge_index.size(1))[:11000]
noise = torch.randint(0,num_items,(2000,))

pos_edges = test_edge_index[:,pos_edges_index]
neg_edges = torch.randint(0,num_items,(pos_edges_index.size(0),))

def cal_cos_sim(users,items,u_idx, i_idx, eps=1e-8, CHUNK_SIZE=1000000):
        user_feature = users
        item_feature = items
        L = u_idx.shape[0]
        sims = torch.zeros(L, dtype=user_feature.dtype).to(world.device)
        for idx in range(0, L, CHUNK_SIZE):
            a_batch = torch.index_select(user_feature, 0, u_idx[idx:idx + CHUNK_SIZE])
            b_batch = torch.index_select(item_feature, 0, i_idx[idx:idx + CHUNK_SIZE])
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims.sigmoid()

sims1 = cal_cos_sim(users,items,pos_edges[0][:10000],torch.cat([pos_edges[1][:8000],noise]))
sims2 = cal_cos_sim(users,items,pos_edges[0],neg_edges)
np_array1 = sims1.detach().cpu().numpy()
np_array2 = sims2.detach().cpu().numpy()
np.save('LightGCN-clean.npy',np_array1)
np.save('LightGCN-noisy.npy',np_array2)
print(sims1)
print(sims2)



