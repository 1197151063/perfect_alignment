"""
The implementation of NR-GCF model
We also offer our re-implementation of LightGCN with PyG
"""

from torch import nn,Tensor
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import torch
import common
from common import cprint
args = common.args
device = common.device

"""
Input for all the models:
    num_users:Number of user nodes;
    num_items:Number of item nodes;
    embedding_dim:Dimention of user and item embedding
    K:layer of GCN
    bipartite_graph:User-Item interaction graph R
    edge_index:Adjancency matrix A: |0   R|
                                    |R^T 0|
"""

class LightGCN(MessagePassing):
    def __init__(self,num_users,
                 num_items,
                 embedding_dim=args.recdim,
                 K=args.layer,
                 add_self_loops=False,
                 edge_index:SparseTensor=None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = self.num_users + self.num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.f = nn.Sigmoid()
        self.add_self_loops = add_self_loops
        self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=self.num_items,embedding_dim=self.embedding_dim)
        nn.init.normal_(self.user_emb.weight,std=0.1)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        self.edge_index = gcn_norm(edge_index)
        cprint("LightGCN will use normal distribution initilizer")

    def computer(self):
        edge_index_norm = self.edge_index
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb,items_emb])
        embs = [all_emb]
        for i in range(self.K):
            all_emb = self.propagate(edge_index_norm,x=all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs,dim=1)
        emb_final = torch.mean(embs,dim=1)
        users,items = torch.split(emb_final,[self.num_users,self.num_items])
        return users,items

    def forward(self,users,items):
        all_users,all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb,items_emb)
        gamma = torch.sum(inner_pro,dim=1)
        return gamma
    
    """
    We offer a numpy version of dot product calculation, 
    since torch.matmul is slow and memory costly
    """
    def getUsersRating(self,users):
        all_users,all_items = self.computer()
        #============THE NUMPY VERSION=============
        # users_emb = all_users[users.long()].cpu().detach().numpy()
        # items_emb = all_items.cpu().detach().numpy()
        # rating = self.f(torch.tensor(np.matmul(users_emb,items_emb.T)))
        #==========================================
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(users_emb @ items_emb.t())
        return rating

    def getEmbedding(self,users,pos_items,neg_items):
        all_users,all_items = self.computer()
        users_emb = all_users[users]
        pos_item_emb = all_items[pos_items]
        neg_item_emb = all_items[neg_items]
        user_0 = self.user_emb(users)
        pos_item_0 = self.item_emb(pos_items)
        neg_item_0 = self.item_emb(neg_items)
        return users_emb,pos_item_emb,neg_item_emb,user_0,pos_item_0,neg_item_0
    
    def bpr_loss(self,users,pos,neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)

class NR_GCF(MessagePassing):
    def __init__(self,num_users,num_items,embedding_dim = args.recdim, 
                 K = args.layer,add_self_loops = False,
                 edge_index:SparseTensor=None,bipartite:SparseTensor=None):
        super().__init__()
        self.edge_index = edge_index
        self.device = device
        self.num_users,self.num_items = num_users,num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.lambda1 = 0.8
        self.norm_weight = args.norm_weight
        self.normal_weight = args.normal_weight
        self.f = nn.Sigmoid()
        self.add_self_loops = add_self_loops
        #E^0 & I^0 -- initial emb
        self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=self.num_items,embedding_dim=self.embedding_dim)
        self.bipartite = bipartite
        nn.init.normal_(self.user_emb.weight,std=0.1)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        cprint("NR-GCF will use normal distribution initilizer")

    """
    user shape: [n * 64]
    item shape: [m * 64] 
    dim = 1 means every user
    dim = 0 means every feature
    """
    
    def norm(self,x):
        user,item = torch.split(x,[self.num_users,self.num_items])
        user_norm = (1e-6 + user.pow(2).sum(dim=1).mean()).sqrt()
        item_norm = (1e-6 + item.pow(2).sum(dim=1).mean()).sqrt()
        user = user  / (item_norm)
        item = item / (user_norm)
        x = torch.cat([user,item])
        return x

    """
    E^k+1 = (1-w)E^k + (w)Norm_signal
    """
    def computer(self):
        edge_index_norm = gcn_norm(self.edge_index)
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb,items_emb])
        embs = [all_emb]
        x = all_emb
        for i in range(self.K):
            x = self.propagate(edge_index=edge_index_norm,x=x)
            norm_signal = self.norm(x)
            x = self.norm_weight * norm_signal + self.normal_weight * x
            embs.append(x)
        embs = torch.stack(embs,dim=1)
        emb_final = torch.mean(embs,dim=1)
        users,items = torch.split(emb_final,[self.num_users,self.num_items])
        return users,items

    def forward(self,users,items):
        all_users,all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb,items_emb)
        gamma = torch.sum(inner_pro,dim=1)
        return gamma
    
    def getUsersRating(self,users):
        all_users,all_items = self.computer()
        #============THE NUMPY VERSION=============
        # users_emb = all_users[users.long()].cpu().detach().numpy()
        # items_emb = all_items.cpu().detach().numpy()
        # rating = self.f(torch.tensor(np.matmul(users_emb,items_emb.T)))
        #==========================================
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self,users,pos_items,neg_items):
        all_users,all_items = self.computer()
        users_emb = all_users[users]
        pos_item_emb = all_items[pos_items]
        neg_item_emb = all_items[neg_items]
        user_0 = self.user_emb(users)
        pos_item_0 = self.item_emb(pos_items)
        neg_item_0 = self.item_emb(neg_items)
        return users_emb,pos_item_emb,neg_item_emb,user_0,pos_item_0,neg_item_0
    
    def bpr_loss(self,users,pos,neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb,pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)
