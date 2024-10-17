from torch import nn,Tensor,optim,LongTensor
from torch_geometric.utils import degree
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.nn.aggr import Aggregation
from gtn_propagation import GeneralPropagation
from torch_geometric.nn.conv import MessagePassing
from recbole.model.init import xavier_normal_initialization
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import torch
import world
import utils
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import negative_sampling,dense_to_sparse,to_scipy_sparse_matrix
from tqdm import tqdm
from torch_scatter import scatter,scatter_max,scatter_softmax,scatter_add
config = world.config
device = world.device
import random
"""
define Recmodels here:
Already implemented : [MF-BPR,NGCF,LightGCN,DGCF,GTN,RGCF,Ours]
"""
"""
General GNN based RecModel
"""
class RecModel(MessagePassing):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 config,
                 edge_index:LongTensor):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.config = config
        self.f = nn.Sigmoid()


    def get_sparse_graph(self,
                         edge_index,
                         use_value=False,
                         value=None):
        num_users = self.num_users
        num_nodes = self.num_nodes
        r,c = edge_index
        row = torch.cat([r , c + num_users])
        col = torch.cat([c + num_users , r])
        if use_value:
            value = torch.cat([value,value])
            return SparseTensor(row=row,col=col,value=value,sparse_sizes=(num_nodes,num_nodes))
        else:
            return SparseTensor(row=row,col=col,sparse_sizes=(num_nodes,num_nodes))
    
    def get_embedding(self):
        raise NotImplementedError
    
    def forward(self,
                edge_label_index:Tensor):
        out = self.get_embedding()
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        out_src = out_u[edge_label_index[0]]
        out_dst = out_i[edge_label_index[1]]
        out_dst_neg = out_i[edge_label_index[2]]
        return (out_src * out_dst).sum(dim=-1),(out_src * out_dst_neg).sum(dim=-1)
    
    def link_prediction(self,
                        src_index:Tensor=None,
                        dst_index:Tensor=None):
        out = self.get_embedding()
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        if src_index is None:
            src_index = torch.arange(self.num_users).long()
        if dst_index is None:
            dst_index = torch.arange(self.num_items).long()
        out_src = out_u[src_index]
        out_dst = out_i[dst_index]
        pred = out_src @ out_dst.t()
        return pred
    
    def recommendation_loss(self,
                            pos_rank,
                            neg_rank,
                            edge_label_index):
        rec_loss = torch.nn.functional.softplus(neg_rank - pos_rank).mean()
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        embedding = torch.cat([user_emb[edge_label_index[0]],
                               item_emb[edge_label_index[1]],
                               item_emb[edge_label_index[2]]])
        regularization = self.config['decay'] * (1/2) * embedding.norm(p=2).pow(2)
        regularization = regularization / pos_rank.size(0)
        return rec_loss , regularization
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)
    
class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(
            in_features=in_dim, out_features=out_dim
        )

    def forward(self, lap_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat

        x = matmul(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2
    
class MF(nn.Module):
    def __init__(self,num_users,num_items,config=config):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = 64
        self.K = config['K']
        self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=num_items,embedding_dim=self.embedding_dim)
        self.f = nn.Sigmoid()
        nn.init.normal_(self.user_emb.weight,std=0.1)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        world.cprint("BPR-MF will use normal distribution initilizer")
    
    def forward(self,users,items):
        """
        We adopt BPR as loss function
        """
        users_emb = self.user_emb.weight[users]
        items_emb = self.item_emb.weight[items] 
        inner_pro = torch.mul(users_emb,items_emb)
        gamma = torch.sum(inner_pro,dim=1)
        return gamma

    def getUsersRating(self,users):
        all_users,all_items = self.user_emb.weight,self.item_emb.weight
        users_emb = all_users[users.long()]
        items_emb = all_items
        #torch.matmul is memory costly
        rating = self.f(users_emb @ items_emb.t())
        return rating
    
    def bpr_loss(self,users,pos,neg):
        users_emb = self.user_emb.weight[users]
        pos_emb = self.item_emb.weight[pos]
        neg_emb = self.item_emb.weight[neg]
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) +
                         neg_emb.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss,reg_loss


"""
Note that the code here is different from the Official Impl
BUT consist with NGCF paper setting
So results may be different from the paper statistics
"""
class NGCF(MessagePassing):
    def __init__(self,num_users,num_items,K = config['K'],edge_index:SparseTensor=None):
        super().__init__()
        self.num_users,self.num_items = num_users,num_items
        self.edge_index_norm = self.NGCF_norm_rw(edge_index)
        # self.edge_index_norm = self.pyg_sparse_to_torch_sparse(self.edge_index_norm)
        self.embedding_dim = 64
        self.K = K
        # self.dropout_list = nn.ModuleList()
        self.weight_size = [self.embedding_dim] * (self.K + 1)
        self.f = nn.Sigmoid()
        self.mess_dropout = [0.1] * self.K
        #E^0 & I^0 -- initial emb
        self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=self.num_items,embedding_dim=self.embedding_dim)
        self.GNNlayers = torch.nn.ModuleList()
        for i in range(self.K):
            self.GNNlayers.append(BiGNNLayer(self.weight_size[i],self.weight_size[i+1]))
        self.apply(xavier_normal_initialization)
        world.cprint("NGCF will use normal distribution initilizer")

    
    """
    The pyg version of the origin normalization of NGCF as A_hat = D^{-1}A
    The reported sys_norm version A_hat = D^{-0.5}AD^{-0.5}
    """
    def NGCF_norm_rw(self,edge_index:SparseTensor):
        num_nodes = self.num_users + self.num_items
        edge_indices = torch.stack([edge_index.storage.row(),edge_index.storage.col()])
        edge_indices,_ = pyg_utils.add_self_loops(edge_indices,num_nodes=num_nodes)
        r,c = edge_indices
        deg = pyg_utils.degree(r,num_nodes=num_nodes,dtype=torch.float32)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm_edge_weight = deg_inv[r]
        return SparseTensor(row=r,col=c,value=norm_edge_weight,sparse_sizes=(num_nodes,num_nodes))

    def NGCF_norm_sys(self,edge_index:SparseTensor):
        return gcn_norm(edge_index,add_self_loops=False)
    
    def pyg_sparse_to_torch_sparse(self,edge_index:SparseTensor):
        num_nodes = self.num_users + self.num_items
        edge_indices = torch.stack([edge_index.storage.row(),edge_index.storage.col()])
        value = edge_index.storage.value()
        return torch.sparse.FloatTensor(edge_indices,value,torch.Size([num_nodes,num_nodes]))
    
    def to_pyg_sparse(self,edge_index):
        num_nodes = self.num_items + self.num_users
        edge_indices = edge_index._indices()
        val = edge_index._values()
        r = edge_indices[0]
        c = edge_indices[1]
        return SparseTensor(row=r,col=c,value=val,sparse_sizes=(num_nodes,num_nodes))
    """
    The propagation of NGCF is:
    E^(k+1) = (L + I)E^(k)W_(1)^(k) + LE^(k)W_(2)^(k)
    L = D^(-0.5)AD^(-0.5)
    """
    def computer(self):
        edge_index_norm = self.edge_index_norm
        # edge_index_norm = self.sparse_drop(edge_index_norm)
        users_emb = self.user_emb.weight    
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb,items_emb])
        embs = [all_emb]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(edge_index_norm, features=all_emb)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(0.1)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embs += [all_embeddings] 
        embs = torch.cat(embs,dim=1)
        users,items = torch.split(embs,[self.num_users,self.num_items])
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
        users_emb = all_users[users.long()]
        items_emb = all_items
        #torch.matmul is memory costly  
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
class AttentionConv(MessagePassing):
    def __init__(self, sigma):
        super(AttentionConv, self).__init__(aggr='add')  # Use 'add' aggregation.
        self.sigma = sigma

    def forward(self, x, edge_index):
        # x: Node embeddings (shape: [N, F])
        # edge_index: Edge indices (shape: [2, E])

        # Normalize embeddings to unit length.
        x = F.normalize(x, p=2, dim=-1)

        # Compute similarity (dot product since embeddings are normalized).
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]

        # Compute squared Euclidean distance using cosine similarity.
        cos_sim = torch.sum(x_i * x_j, dim=-1)  # Shape: [E]
        sq_dist = 2 - 2 * cos_sim  # Since ||x_i - x_j||^2 = 2 - 2 * cos_sim

        # Compute Gaussian kernel weights.
        alpha = torch.exp(-sq_dist / (self.sigma ** 2))

        # Normalize attention coefficients.
        alpha = self.edge_softmax(row, alpha)

        # Multiply source embeddings by attention coefficients.
        out = self.propagate(edge_index, x=x, alpha=alpha)

        return out

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

    def edge_softmax(self, index, alpha):
        # Compute softmax over edges for each node.
        alpha_exp = alpha
        alpha_sum = scatter_add(alpha_exp, index, dim=0)
        alpha_norm = alpha_exp / (alpha_sum[index] + 1e-16)
        return alpha_norm

    def update(self, aggr_out):
        # No need to normalize again if not necessary.
        return aggr_out
    
class LightGCN(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            config=config,
            edge_index=edge_index
        )
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['latent_dim_rec'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['latent_dim_rec'])
        nn.init.normal_(self.user_emb.weight,std=0.1)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        self.K = config['K']
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(edge_index)
        self.alpha= 1./ (1 + self.K)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K + 1
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        print('Go LightGCN')
        print(f"params settings: \n emb_size:{config['latent_dim_rec']}\n L2 reg:{config['decay']}\n layer:{self.K}")

    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out

    def instance_loss(self,edge_label_index):
        out = self.get_embedding()
        users,items = torch.split(out,[self.num_users,self.num_items])
        user_emb = users[edge_label_index[0]]
        item_pos = items[edge_label_index[1]]
        item_neg = items[edge_label_index[2]]
        return ((user_emb * item_pos).sum(dim=-1) - (user_emb * item_neg).sum(dim=-1)).sigmoid()

class AlignGCN(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            config=config,
            edge_index=edge_index
        )
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['latent_dim_rec'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['latent_dim_rec'])
        nn.init.xavier_uniform_(self.user_emb.weight,gain=2)
        nn.init.xavier_uniform_(self.item_emb.weight,gain=2)
        self.K = config['K']
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(edge_index)
        self.alpha= 1./ (1 + self.K)
        self.au = config['au']
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K + 1
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        print('Encoder: LightGCN')
        print(f"params settings: \n emb_size:{config['latent_dim_rec']}\n L2 reg:{config['decay']}\n layer:{self.K}")

    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out
    
    def get_embedding_pure(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        return x
    
    def get_embedding_norm(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            x = self.norm(x)
            out = out + x * self.alpha[i + 1]
        return out
    
    def get_embedding_wo_1layer(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * 0
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out = out + x * self.alpha[i]
        return out 

    def get_embedding_wo_1layer_norm(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * 0
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            x = self.norm(x)
            out = out + x * self.alpha[i]
        return out 

    def alignment(self,x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(dim=1).pow(2).mean()

    def alignment_loss(self,edge_label_index):
        out = self.get_embedding_norm()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        batch_x_u,batch_x_i = x_u[edge_label_index[0]],x_i[edge_label_index[1]]
        return self.alignment(batch_x_u,batch_x_i)
    
    def uniformity_loss(self,edge_label_index):
        out = self.get_embedding_norm()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(torch.cat([edge_label_index[1],edge_label_index[2]]))
        batch_x_u,batch_x_i = x_u[u_idx],x_i[i_idx]
        return  self.au * (self.uniformity(batch_x_u) + self.uniformity(batch_x_i))

    def pair_align(self,edge_label_index):
        out = self.get_embedding()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        users = x_u[edge_label_index[0]]
        items = x_i[edge_label_index[1]]
        dot_product = torch.sum(users * items, dim=-1)
        sigmoid_scores = torch.sigmoid(dot_product)
        log_loss = -torch.log(sigmoid_scores + 1e-8)
        return log_loss.sum()

    def entropy_based_uniformity(self,x):
        x = F.normalize(x, dim=-1)
        dist_matrix = torch.cdist(x, x, p=2)
        p_dist = torch.exp(-dist_matrix)
        entropy = -torch.mean(p_dist * torch.log(p_dist + 1e-8))
        return entropy
    
    def pair_entropy(self,edge_label_index):
        out = self.get_embedding()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        users = x_u[edge_label_index[0]]
        items = x_i[edge_label_index[1]]
        user_u = self.entropy_based_uniformity(users)
        item_u = self.entropy_based_uniformity(items)
        return 15000 * (user_u + item_u)
    def contrastive_uniformity(self,x, temperature=0.5):
        # 归一化
        x = F.normalize(x, dim=-1)
        
        # 计算pairwise相似度（内积）
        similarity_matrix = torch.matmul(x, x.t()) / temperature
        
        # 计算对比学习损失
        logits = torch.exp(similarity_matrix)
        loss = -torch.log(logits / torch.sum(logits, dim=-1, keepdim=True) + 1e-8).mean()
        
        return loss
    def pair_contrast(self,edge_label_index):
        out = self.get_embedding()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        users = x_u[edge_label_index[0]]
        items = x_i[edge_label_index[1]]
        return self.contrastive_uniformity(users) + self.contrastive_uniformity(items)     
    def pair_norm(self,edge_label_index):
        out = self.get_embedding()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        users = x_u[edge_label_index[0]]
        items = x_i[edge_label_index[1]]
        user_dist = torch.cdist(users, users, p=2)  
        user_sim = F.cosine_similarity(users.unsqueeze(1), users.unsqueeze(0), dim=-1)
        item_dist = torch.cdist(items, items, p=2)  
        item_sim = F.cosine_similarity(items.unsqueeze(1), items.unsqueeze(0), dim=-1)        
        user_dist = user_dist + torch.eye(user_dist.size(0)).cuda() * 1e-6
        item_dist = item_dist + torch.eye(item_dist.size(0)).cuda() * 1e-6
        user_sim = torch.clamp(user_sim, -1.0, 1.0)  
        user_dist = torch.clamp(user_dist, min=1e-6)
        item_sim = torch.clamp(item_sim, -1.0, 1.0)  
        item_dist = torch.clamp(item_dist, min=1e-6)
        user_uniformity = torch.sum((1 - user_sim) / user_dist)
        item_uniformity = torch.sum((1 - item_sim) / item_dist)
        uniformity_loss = 2 * user_uniformity + 2 * item_uniformity
        return uniformity_loss

    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()) ** (1/2)
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()) ** (1/2)
        if users_norm < 1:
            users_norm = 1 
        if items_norm < 1:
            items_norm = 1
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x

    def get_shuffle_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = [x]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            r_noise = torch.rand_like(x).cuda()
            x = x + torch.sign(x) * F.normalize(r_noise,dim=-1) * 0.1
            out.append(x)
        out = torch.stack(out,dim=1)
        out = torch.mean(out,dim=1)
        return out
    
    def cal_minimum_loss(self,edge_label_index):
        out = self.get_embedding()
        out_aug = self.get_shuffle_embedding()
        users,items = torch.split(out,[self.num_users,self.num_items])
        user_emb = users[edge_label_index[0]]
        item_pos = items[edge_label_index[1]]
        alignment_loss = self.alignment(user_emb,item_pos)
        alignment_ui = self.alignment(out,out_aug)
        align = max(alignment_loss - alignment_ui + 0.2,0)
        return align

    def intra_aggregation(self,x,y):
        x = F.normalize(x,dim=-1)
        y = F.normalize(y,dim=-1)
        cos_sim = F.cosine_similarity(x,y)
        return (1-cos_sim).mean()
    
    def seperation_loss(self,x):
        x = F.normalize(x,dim=-1)
        x_c = x.mean()
        cos_sim = F.cosine_similarity(x,x_c)
        return -cos_sim.mean()

    def intra_and_seperation(self,edge_label_index):
        out = self.get_embedding()
        user_emb,item_emb = torch.split(out,[self.num_users,self.num_items])
        users,items = user_emb[edge_label_index[0]],item_emb[edge_label_index[1]]
        alignment_loss = self.intra_aggregation(users,items)
        seperation_loss = self.seperation_loss(users) + self.seperation_loss(items)
        return alignment_loss,seperation_loss

    
    def instance_loss(self,edge_label_index):
        out = self.get_embedding()
        users,items = torch.split(out,[self.num_users,self.num_items])
        user_emb = users[edge_label_index[0]]
        item_pos = items[edge_label_index[1]]
        item_neg = items[edge_label_index[2]]
        return ((user_emb * item_pos).sum(dim=-1) - (user_emb * item_neg).sum(dim=-1)).sigmoid()
    
    def decorelation_loss(self,x):
        size, dim = x.size()
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (size - 1)
        diag = torch.eye(dim, device=x.device)
        cov_diff = cov - diag
        return torch.norm(cov_diff, p='fro') ** 2
    
    def cal_decorelation_loss(self,edge_label_index):
        out = self.get_embedding_norm()
        users = torch.unique(edge_label_index[0])
        items = torch.unique(torch.cat([edge_label_index[1],edge_label_index[2]]))
        out = torch.cat([out[users],out[items + self.num_users]])
        return self.decorelation_loss(out) * 0.01
    
class NR_GCF(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            config=config,
            edge_index=edge_index
        )
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['latent_dim_rec'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['latent_dim_rec'])
        nn.init.normal_(self.user_emb.weight,std=0.15)
        nn.init.normal_(self.item_emb.weight,std=0.15)
        self.K = config['K']
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(edge_index)
        self.alpha= 1./ (1 + self.K)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K + 1
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        print('Go NRGCF')
        print(f"params settings: \n emb_size:{config['latent_dim_rec']}\n L2 reg:{config['decay']}\n layer:{self.K}")

    def norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()).sqrt()
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()).sqrt()
        if users_norm > 1:
            users_norm = 1 
        if items_norm > 1:
            items_norm = 1
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x
    
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            x = self.norm(x)
            out = out + x * self.alpha[i + 1]
        return out    

class SGL(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config,
                 pre_users=None,
                 pre_items=None):
        super().__init__(num_users=num_users,
                         num_items=num_items,
                         edge_index=edge_index,
                         config=config)
        self.K = config['K']
        self.num_interactions = edge_index.size(1)
        self.edge_index = self.get_sparse_graph(edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(self.edge_index)        
        self.alpha= 1./ (1 + self.K)
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['latent_dim_rec'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['latent_dim_rec'])
        self.pre_users = pre_users
        self.pre_items = pre_items
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K + 1
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        self.ssl_tmp = config['ssl_tmp']
        self.ssl_decay = config['ssl_decay']    
        print('Go SGL')
        print(f"params settings: \n emb_size:{config['latent_dim_rec']}\n L2 reg:{config['decay']}\n layer:{self.K}")
        print(f" ssl_tmp:{config['ssl_tmp']}\n ssl_decay:{config['ssl_decay']}\n graph aug type: edge drop")

    
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out
    
    def forward(self,
                edge_label_index:Tensor):
        out = self.get_embedding()
        # pred_pos_sim = self.pre_users[edge_label_index[0]] @ self.pre_items[edge_label_index[1]].t()
        # pred_pos_sim = self.f(pred_pos_sim)
        # pred_pos_sim = torch.diag(pred_pos_sim)
        # pred_neg_sim = self.pre_users[edge_label_index[0]] @ self.pre_items[edge_label_index[2]].t()
        # pred_neg_sim = self.f(pred_neg_sim)
        # pred_neg_sim = torch.diag(pred_neg_sim)
        out_u,out_i = torch.split(out,[self.num_users,self.num_items])
        out_src = out_u[edge_label_index[0]]
        out_dst = out_i[edge_label_index[1]]
        out_dst_neg = out_i[edge_label_index[2]]
        return (out_src * out_dst).sum(dim=-1) ,(out_src * out_dst_neg).sum(dim=-1) 

    def get_ssl_embedding(self,edge_index):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.propagate(edge_index=edge_index,x=x)
            out = out + x * self.alpha[i + 1]
        return out

    def min_max_norm(self,x:Tensor):
        return (x - x.min())/(x.max()-x.min())

    
    # def ssl_forward(self,
    #                 edge_index1,
    #                 edge_index2,
    #                 edge_label_index:LongTensor):
    #     info_out1 = self.get_ssl_embedding(edge_index1)
    #     info_out2 = self.get_ssl_embedding(edge_index2)
    #     info_out_u_1,info_out_i_1 = torch.split(info_out1,[self.num_users,self.num_items])
    #     info_out_u_2,info_out_i_2 = torch.split(info_out2,[self.num_users,self.num_items])
    #     u_idx = torch.unique(edge_label_index[0])
    #     i_idx = torch.unique(edge_label_index[1])
    #     info_out_u1 = info_out_u_1[u_idx]
    #     info_out_u2 = info_out_u_2[u_idx]
    #     info_out_i1 = info_out_i_1[i_idx]
    #     info_out_i2 = info_out_i_2[i_idx]
    #     info_out_u1 = F.normalize(info_out_u1,dim=1)
    #     info_out_u2 = F.normalize(info_out_u2,dim=1)
    #     info_pos_user = (info_out_u1 * info_out_u2).sum(dim=1)/ self.ssl_tmp
    #     info_pos_user = torch.exp(info_pos_user)
    #     info_neg_user = (info_out_u1 @ info_out_u2.t())/ self.ssl_tmp
    #     info_neg_user = torch.exp(info_neg_user)
    #     info_neg_user = torch.sum(info_neg_user,dim=1,keepdim=True)
    #     info_neg_user = info_neg_user.T
    #     ssl_logits_user = -torch.log(info_pos_user / info_neg_user).mean()
    #     info_out_i1 = F.normalize(info_out_i1,dim=1)
    #     info_out_i2 = F.normalize(info_out_i2,dim=1)
    #     info_pos_item = (info_out_i1 * info_out_i2).sum(dim=1)/ self.ssl_tmp
    #     info_neg_item = (info_out_i1 @ info_out_i2.t())/ self.ssl_tmp
    #     info_pos_item = torch.exp(info_pos_item)
    #     info_neg_item = torch.exp(info_neg_item)
    #     info_neg_item = torch.sum(info_neg_item,dim=1,keepdim=True)
    #     info_neg_item = info_neg_item.T
    #     ssl_logits_item = -torch.log(info_pos_item / info_neg_item).mean()
    #     return ssl_logits_user,ssl_logits_item
    def focal_ssl_loss(self,
                       edge_index1,
                       edge_index2,
                       edge_label_index):
        info_out1 = self.get_ssl_embedding(edge_index1)
        info_out2 = self.get_ssl_embedding(edge_index2)
        info_out_u_1,info_out_i_1 = torch.split(info_out1,[self.num_users,self.num_items])
        info_out_u_2,info_out_i_2 = torch.split(info_out2,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        user_sim = self.min_max_norm(torch.diag(self.pre_users[u_idx] @ self.pre_users[u_idx].t()))
        item_sim = self.min_max_norm(torch.diag(self.pre_items[i_idx] @ self.pre_items[i_idx].t()))
        user_mask = user_sim > 0.5
        item_mask = item_sim > 0,5
        user_sim = torch.where(user_mask,user_sim - 0.3,user_sim)
        item_sim = torch.where(item_mask,item_sim - 0.3,item_sim)
        info_out_u1 = info_out_u_1[u_idx]
        info_out_u2 = info_out_u_2[u_idx]
        info_out_i1 = info_out_i_1[i_idx]
        info_out_i2 = info_out_i_2[i_idx]
        info_out_u1 = F.normalize(info_out_u1,dim=1)
        info_out_u2 = F.normalize(info_out_u2,dim=1)
        # info_out_u_2 = F.normalize(info_out_u_2,dim=1)
        # info_out_i_2 = F.normalize(info_out_i_2,dim=1)
        info_pos_user = (info_out_u1 * info_out_u2).sum(dim=1)/ self.ssl_tmp
        info_pos_user = torch.exp(info_pos_user)
        info_neg_user = (info_out_u1 @ info_out_u2.t())/ self.ssl_tmp
        info_neg_user = torch.exp(info_neg_user)
        # info_neg_user[user_diag_indices,user_diag_indices] = 1e-8
        info_neg_user = torch.sum(info_neg_user,dim=1,keepdim=True)
        info_neg_user = info_neg_user.T * user_sim 
        ssl_logits_user = -torch.log(info_pos_user / info_neg_user).mean()
        info_out_i1 = F.normalize(info_out_i1,dim=1)
        info_out_i2 = F.normalize(info_out_i2,dim=1)
        info_pos_item = (info_out_i1 * info_out_i2).sum(dim=1)/ self.ssl_tmp
        info_neg_item = (info_out_i1 @ info_out_i2.t())/ self.ssl_tmp
        info_pos_item = torch.exp(info_pos_item)
        info_neg_item = torch.exp(info_neg_item)
        # info_neg_item[item_diag_indices,item_diag_indices] = 1e-8
        info_neg_item = torch.sum(info_neg_item,dim=1,keepdim=True)
        info_neg_item = info_neg_item.T * item_sim 
        ssl_logits_item = -torch.log(info_pos_item / info_neg_item).mean()
    #     return ssl_logits_user,ssl_logits_item
        # v1 = torch.cat([view1[u_idx],view1[i_idx + self.num_users]])
        # v2 = torch.cat([view2[u_idx],view2[i_idx + self.num_users]])
        # info_loss = self.ssl_decay * utils.InfoNCE(v1,v2,self.ssl_tmp)
        return self.ssl_decay * (ssl_logits_user + ssl_logits_item)   

    def ssl_loss(self,
                    edge_index1,
                    edge_index2,
                    edge_label_index):
        info_out1 = self.get_ssl_embedding(edge_index1)
        info_out2 = self.get_ssl_embedding(edge_index2)
        info_out_u_1,info_out_i_1 = torch.split(info_out1,[self.num_users,self.num_items])
        info_out_u_2,info_out_i_2 = torch.split(info_out2,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        info_out_u1 = info_out_u_1[u_idx]
        info_out_u2 = info_out_u_2[u_idx]
        info_out_i1 = info_out_i_1[i_idx]
        info_out_i2 = info_out_i_2[i_idx]
        info_out_u1 = F.normalize(info_out_u1,dim=1)
        info_out_u2 = F.normalize(info_out_u2,dim=1)
        info_out_u_2 = F.normalize(info_out_u_2,dim=1)
        info_out_i_2 = F.normalize(info_out_i_2,dim=1)
        info_pos_user = (info_out_u1 * info_out_u2).sum(dim=1)/ self.ssl_tmp
        info_pos_user = torch.exp(info_pos_user)
        info_neg_user = (info_out_u1 @ info_out_u_2.t())/ self.ssl_tmp
        info_neg_user = torch.exp(info_neg_user)
        info_neg_user = torch.sum(info_neg_user,dim=1,keepdim=True)
        info_neg_user = info_neg_user.T
        ssl_logits_user = -torch.log(info_pos_user / info_neg_user).mean()
        info_out_i1 = F.normalize(info_out_i1,dim=1)
        info_out_i2 = F.normalize(info_out_i2,dim=1)
        info_pos_item = (info_out_i1 * info_out_i2).sum(dim=1)/ self.ssl_tmp
        info_neg_item = (info_out_i1 @ info_out_i_2.t())/ self.ssl_tmp
        info_pos_item = torch.exp(info_pos_item)
        info_neg_item = torch.exp(info_neg_item)
        info_neg_item = torch.sum(info_neg_item,dim=1,keepdim=True)
        info_neg_item = info_neg_item.T
        ssl_logits_item = -torch.log(info_pos_item / info_neg_item).mean()
    #     return ssl_logits_user,ssl_logits_item
        # v1 = torch.cat([view1[u_idx],view1[i_idx + self.num_users]])
        # v2 = torch.cat([view2[u_idx],view2[i_idx + self.num_users]])
        # info_loss = self.ssl_decay * utils.InfoNCE(v1,v2,self.ssl_tmp)
        return self.ssl_decay * (ssl_logits_user + ssl_logits_item)
    
    def bpr_loss(self,pos_rank,neg_rank):
        return F.softplus(neg_rank - pos_rank).mean()
    
    def L2_reg(self,edge_label_index):
        u_idx,i_idx_pos,i_idx_neg = edge_label_index
        userEmb0 = self.user_emb.weight[u_idx]
        posEmb0 = self.item_emb.weight[i_idx_pos]
        negEmb0 = self.item_emb.weight[i_idx_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / edge_label_index.size(1)
        regularization = self.config['decay'] * reg_loss
        return regularization
    
class SimGCL(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(num_users=num_users,
                         num_items=num_items,
                         edge_index=edge_index,
                         config=config
                         )
        self.K = config['K']
        self.num_interactions = edge_index.size(1)
        self.train_edge_index = edge_index
        self.edge_index = self.get_sparse_graph(edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(self.edge_index)        
        self.alpha= 1./ (self.K)
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['latent_dim_rec'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['latent_dim_rec'])
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K))
        self.ssl_tmp = config['ssl_tmp']
        self.ssl_decay = config['ssl_decay']
        self.eps = config['epsilon']
        print('Go backbone SimGCL')
        print(f"params settings: \n L2 reg:{config['decay']}\n layer:{self.K}")
        print(f" ssl_tmp:{config['ssl_tmp']}\n ssl_decay:{config['ssl_decay']}\n noise_bias:{config['epsilon']}")
    
    def norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()).sqrt()
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()).sqrt()
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x
    
    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = []
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            out.append(x)
        out = torch.stack(out,dim=1)
        out = torch.mean(out,dim=1)
        return out
    
    def get_shuffle_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = []
        for i in range(self.K):
            x = self.propagate(edge_index=self.edge_index,x=x)
            r_noise = torch.rand_like(x).cuda()
            x = x + torch.sign(x) * F.normalize(r_noise,dim=-1) * self.eps
            out.append(x)
        out = torch.stack(out,dim=1)
        out = torch.mean(out,dim=1)
        return out
    
    def ssl_loss(self,edge_label_index):
        u_idx,i_idx = edge_label_index[0],edge_label_index[1]
        view1 = self.get_shuffle_embedding()
        view2 = self.get_shuffle_embedding()
        info_out_u_1,info_out_i_1 = torch.split(view1,[self.num_users,self.num_items])
        info_out_u_2,info_out_i_2 = torch.split(view2,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        user_cl_loss = utils.InfoNCE(info_out_u_1[u_idx], info_out_u_2[u_idx], 0.2)
        item_cl_loss = utils.InfoNCE(info_out_i_1[i_idx], info_out_i_2[i_idx], 0.2)
        return self.ssl_decay * (user_cl_loss + item_cl_loss)    
    
    def focal_ssl_loss(self,
                       edge_label_index):
        info_out1 = self.get_shuffle_embedding()
        info_out2 = self.get_shuffle_embedding()
        info_out_u_1,info_out_i_1 = torch.split(info_out1,[self.num_users,self.num_items])
        info_out_u_2,info_out_i_2 = torch.split(info_out2,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(edge_label_index[1])
        info_out_u1 = info_out_u_1[u_idx]
        info_out_u2 = info_out_u_2[u_idx]
        info_out_i1 = info_out_i_1[i_idx]
        info_out_i2 = info_out_i_2[i_idx]
        info_out_u1 = F.normalize(info_out_u1,dim=1)
        info_out_u2 = F.normalize(info_out_u2,dim=1)
        # info_out_u_2 = F.normalize(info_out_u_2,dim=1)
        # info_out_i_2 = F.normalize(info_out_i_2,dim=1)
        info_pos_user = (info_out_u1 * info_out_u2).sum(dim=1)/ self.ssl_tmp
        info_pos_user = torch.exp(info_pos_user)
        info_neg_user = (info_out_u1 @ info_out_u2.t())/ self.ssl_tmp
        info_neg_user = torch.exp(info_neg_user)
        # info_neg_user[user_diag_indices,user_diag_indices] = 1e-8
        info_neg_user = torch.sum(info_neg_user,dim=1,keepdim=True)
        ssl_logits_user = -torch.log(info_pos_user / info_neg_user).mean()
        info_out_i1 = F.normalize(info_out_i1,dim=1)
        info_out_i2 = F.normalize(info_out_i2,dim=1)
        info_pos_item = (info_out_i1 * info_out_i2).sum(dim=1)/ self.ssl_tmp
        info_neg_item = (info_out_i1 @ info_out_i2.t())/ self.ssl_tmp
        info_pos_item = torch.exp(info_pos_item)
        info_neg_item = torch.exp(info_neg_item)
        # info_neg_item[item_diag_indices,item_diag_indices] = 1e-8
        info_neg_item = torch.sum(info_neg_item,dim=1,keepdim=True)
        ssl_logits_item = -torch.log(info_pos_item / info_neg_item).mean()
    #     return ssl_logits_user,ssl_logits_item
        # v1 = torch.cat([view1[u_idx],view1[i_idx + self.num_users]])
        # v2 = torch.cat([view2[u_idx],view2[i_idx + self.num_users]])
        # info_loss = self.ssl_decay * utils.InfoNCE(v1,v2,self.ssl_tmp)
        return self.ssl_decay * (ssl_logits_user + ssl_logits_item) 

    def bpr_loss(self,pos_rank,neg_rank):
        return F.softplus(neg_rank - pos_rank).mean()
    
    def L2_reg(self,edge_label_index):
        u_idx,i_idx_pos,i_idx_neg = edge_label_index
        userEmb0 = self.user_emb.weight[u_idx]
        posEmb0 = self.item_emb.weight[i_idx_pos]
        negEmb0 = self.item_emb.weight[i_idx_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / edge_label_index.size(1)
        regularization = self.config['decay'] * reg_loss
        return regularization

class NormGCN(RecModel):
    def __init__(self,
                 num_users:int,
                 num_items:int,
                 edge_index:LongTensor,
                 config):
        super().__init__(num_users=num_users,
                         num_items=num_items,
                         edge_index=edge_index,
                         config=config
                         )
        self.K = config['K']
        self.num_interactions = edge_index.size(1)
        self.train_edge_index = edge_index
        self.edge_index = self.get_homo_edge_index(self.train_edge_index).to(device)      
        self.alpha= 1./ (self.K)
        self.conv = AttentionConv(sigma=0.5)
        self.user_emb = nn.Embedding(num_embeddings=num_users,
                                     embedding_dim=config['latent_dim_rec'])
        self.item_emb = nn.Embedding(num_embeddings=num_items,
                                     embedding_dim=config['latent_dim_rec'])
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.K
        else:
            self.alpha = torch.tensor([self.alpha] * (self.K + 1))
        self.ssl_tmp = config['ssl_tmp']
        self.ssl_decay = config['ssl_decay']
        self.eps = config['epsilon']
        print('Go backbone SimGCL')
        print(f"params settings: \n L2 reg:{config['decay']}\n layer:{self.K}")
        print(f" ssl_tmp:{config['ssl_tmp']}\n ssl_decay:{config['ssl_decay']}\n noise_bias:{config['epsilon']}")
    
    def get_homo_edge_index(self,edge_index):
        r,c = edge_index
        row = torch.cat([r,c + self.num_users])
        col = torch.cat([c + self.num_users,r])
        return torch.stack([row,col],dim=0)
    
    def compute_edge_weight(self, x):
        # Compute edge weights based on node degrees and beta.
        row, col = self.edge_index
        num_nodes = x.size(0)
        deg = torch.zeros(num_nodes, device=x.device)
        deg = deg.index_add_(0, row, torch.ones_like(row, dtype=x.dtype,device=x.device))

        deg_beta = deg.pow(self.beta)

        deg_row = deg_beta[row]
        deg_col = deg_beta[col]
        edge_weight = 1.0 / (deg_row * deg_col).sqrt()
        return edge_weight
    
    def norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()).sqrt()
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()).sqrt()
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x
    
    def cal_weight(self):
        chunk_size = 4096
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        train_edge_index = self.train_edge_index
        edge_index = self.get_homo_edge_index(self.train_edge_index)
        row,col = edge_index
        sims_sp = torch.zeros_like(edge_index[0])
        for index in range(0,train_edge_index[0].size(0),chunk_size):
            users = x_u[train_edge_index[0][index]]
            items = x_i[train_edge_index[1][index]]
            sims = users @ items
            sims_sp[:index] = sims
        sims_scaled = sims_sp 
        weights = scatter(F.softmax(sims_scaled,dim=0),row,dim=0)
        self.weight = weights

    def get_embedding(self):
        x_u=self.user_emb.weight
        x_i=self.item_emb.weight
        x=torch.cat([x_u,x_i])
        out = x * self.alpha[0]
        for i in range(self.K):
            x = self.conv(x,self.edge_index)
            x = self.norm(x)
            out = out + x * self.alpha[i + 1]
        return out

    def bpr_loss(self,pos_rank,neg_rank):
        return F.softplus(neg_rank - pos_rank).mean()
    
    def L2_reg(self,edge_label_index):
        u_idx,i_idx_pos,i_idx_neg = edge_label_index
        userEmb0 = self.user_emb.weight[u_idx]
        posEmb0 = self.item_emb.weight[i_idx_pos]
        negEmb0 = self.item_emb.weight[i_idx_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2)) / edge_label_index.size(1)
        regularization = self.config['decay'] * reg_loss
        return regularization

    def message(self, x_j, edge_weight):
        # 使用边权重对邻居的消息进行加权
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 更新后的节点表示
        return aggr_out
# class LightGCN(MessagePassing):
#     def __init__(self,num_users,num_items,embedding_dim = config['latent_dim_rec'] , K = config['K'],add_self_loops = False,edge_index:SparseTensor=None):
#         super().__init__()
#         self.num_users,self.num_items = num_users,num_items
#         self.embedding_dim = embedding_dim
#         self.K = K
#         self.f = nn.Sigmoid()
#         self.add_self_loops = add_self_loops
#         #E^0 & I^0 -- initial emb
#         self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
#         self.item_emb = nn.Embedding(num_embeddings=self.num_items,embedding_dim=self.embedding_dim)
#         nn.init.normal_(self.user_emb.weight,std=0.1)
#         nn.init.normal_(self.item_emb.weight,std=0.1)
#         self.edge_index_norm = gcn_norm(edge_index)
#         world.cprint("LightGCN will use normal distribution initilizer")

#     def computer(self):
#         edge_index_norm = self.edge_index_norm
#         users_emb = self.user_emb.weight
#         items_emb = self.item_emb.weight
#         all_emb = torch.cat([users_emb,items_emb])
#         embs = [all_emb]
#         for i in range(self.K):
#             all_emb = self.propagate(edge_index_norm,x=all_emb)
#             embs.append(all_emb)
#         embs = torch.stack(embs,dim=1)
#         emb_final = torch.mean(embs,dim=1)
#         users,items = torch.split(emb_final,[self.num_users,self.num_items])
#         return users,items

#     def forward(self,users,items):
#         all_users,all_items = self.computer()
#         users_emb = all_users[users]
#         items_emb = all_items[items]
#         inner_pro = torch.mul(users_emb,items_emb)
#         gamma = torch.sum(inner_pro,dim=1)
#         return gamma
    
#     def getUsersRating(self,users):
#         all_users,all_items = self.computer()
#         users_emb = all_users[users.long()]
#         items_emb = all_items
#         rating = self.f(users_emb @ items_emb.t())
#         return rating

#     def getEmbedding(self,users,pos_items,neg_items):
#         all_users,all_items = self.computer()
#         users_emb = all_users[users]
#         pos_item_emb = all_items[pos_items]
#         neg_item_emb = all_items[neg_items]
#         user_0 = self.user_emb(users)
#         pos_item_0 = self.item_emb(pos_items)
#         neg_item_0 = self.item_emb(neg_items)
#         return users_emb,pos_item_emb,neg_item_emb,user_0,pos_item_0,neg_item_0
    
#     def bpr_loss(self,users,pos,neg):
#         (users_emb, pos_emb, neg_emb, 
#         userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
#         reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) +
#                          negEmb0.norm(2).pow(2))/float(len(users))
#         pos_scores = torch.mul(users_emb, pos_emb)
#         pos_scores = torch.sum(pos_scores, dim=1) 
#         neg_scores = torch.mul(users_emb, neg_emb)
#         neg_scores = torch.sum(neg_scores, dim=1)
#         loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
#         return loss, reg_loss
    
#     def message(self, x_j: Tensor) -> Tensor:
#         return x_j
    
#     def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#         return matmul(adj_t,x)
    
class GTN(MessagePassing):

    def __init__(self,num_users,num_items,embedding_dim = config['latent_dim_rec'] , 
                 K = config['K'],add_self_loops = False,edge_index:SparseTensor=None,
                 args=None):
        super().__init__()
        self.num_users,self.num_items = num_users,num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.f = nn.Sigmoid()
        self.add_self_loops = add_self_loops
        #E^0 & I^0 -- initial emb
        self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=self.num_items,embedding_dim=self.embedding_dim)
        nn.init.normal_(self.user_emb.weight,std=0.1)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        self.edge_index = edge_index
        self.gp = GeneralPropagation(self.K, args.alpha, cached=True, args=args)
        world.cprint("GTN will use normal distribution initilizer")

    def computer(self):
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb,items_emb])
        embs,_ = self.gp.forward(x=all_emb,edge_index=self.edge_index)
        users,items = torch.split(embs,[self.num_users,self.num_items])
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
    
"""
num_users:Number of user nodes;
num_items:Number of item nodes;
embedding_dim:Dimention of user and item embedding
K:layer of GCN
bipartite_graph:User-Item interaction graph R
edge_index:Adjancency matrix A: |0   R|
                                |R^T 0|
"""
class NRGCF(MessagePassing):
    def __init__(self,num_users,num_items,
                 embedding_dim = config['latent_dim_rec'], 
                 K = config['K'],add_self_loops = False,
                 edge_index:SparseTensor=None,
                 pre_users=None,pre_items=None,
                 scaling_conf=None,
                 adj_mat:SparseTensor=None):
        super().__init__()
        self.edge_index = edge_index
        self.device = world.device
        self.num_users,self.num_items = num_users,num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.scaling_conf = scaling_conf
        self.norm_weight = config['norm_weight']
        self.normal_weight = config['normal_weight']
        self.f = nn.Sigmoid()
        self.add_self_loops = add_self_loops
        self.pre_users = pre_users
        self.pre_items = pre_items
        # self.adj_mat = adj_mat
        # self.sim = self.get_sim_sp(self.pre_users,self.pre_items)
        self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=self.num_items,embedding_dim=self.embedding_dim)
        nn.init.normal_(self.user_emb.weight,std=0.1)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        world.cprint("Our GCN will use normal distribution initilizer")



    """
    user shape: [n * 64]
    item shape: [m * 64] 
    dim = 1 means every user
    dim = 0 means every feature
    """
    def norm(self,x,x_n):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_n,items_n = torch.split(x_n,[self.num_users,self.num_items])
        users_norm = (1e-6 + users.pow(2).sum(dim=1).mean()).sqrt()
        items_norm = (1e-6 + items.pow(2).sum(dim=1).mean()).sqrt()
        users = (self.scaling_conf * users / (items_norm)) + users_n
        items = (self.scaling_conf * items / (users_norm)) + items_n
        x = torch.cat([users,items])
        return x

    def computer(self): 
        edge_index_norm = gcn_norm(self.edge_index)
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb,items_emb])
        embs = [all_emb]
        x = all_emb
        # x = self.dropout(x)
        """
        E^k+1 = AE^k
        E^k+1 = E^k + (AE^k - E^k)
        E^K+1 = E^k + norm ()
        """

        for i in range(self.K):
            x_n = x
            x = self.propagate(edge_index=edge_index_norm,x=x)
            norm_signal = self.norm(x,x_n)
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

    

class RGCF(MessagePassing):
    def __init__(self,num_users,num_items,embedding_dim = config['latent_dim_rec'] , K = config['K'],add_self_loops = False,edge_index:SparseTensor=None,adj_mat:SparseTensor=None):
        super().__init__()
        self.adj_mat = adj_mat
        self.edge_index = edge_index
        self.num_users,self.num_items = num_users,num_items
        self.user_index = adj_mat.storage.row()
        self.item_index = adj_mat.storage.col()
        self.embedding_dim = embedding_dim
        self.K = K
        self.tau = 0.2
        self.f = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.add_self_loops = add_self_loops
        #E^0 & I^0 -- initial emb
        self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=self.num_items,embedding_dim=self.embedding_dim)
        nn.init.normal_(self.user_emb.weight,std=0.1)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        world.cprint("RGCF will use normal distribution initilizer")
        # self.edge_index_denoised = self.graph_denoising_module(self.edge_index)
        # world.cprint("GRAPH AUGMENTATION")
        # self.aug_edge_index = self.graph_augmentation_module(self.denoised_adj)
    """
    Graph Denoising Module
    input: edge_index(origin graph)
    output:edge_index(denoised graph)
    """


    """
    torch.matmul is extreamly memory costly and relatively slow
    we design a special cosine similarity calculation method
    in order to save CUDA memory
    """

    def get_cos_sim_sp(self,a,b,eps=1e-8,CHUNK_SIZE=1000000):
        a_n, b_n = a.norm(dim=1)[:,None],b.norm(dim=1)[:,None]
        a_norm = a / torch.max(a_n,eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n,eps * torch.ones_like(b_n))
        L = len(self.adj_mat.storage.row())
        sims = torch.zeros(L,dtype=a.dtype).to(world.device)
        for idx in range(0,L,CHUNK_SIZE):
            batch_row_index = self.adj_mat.storage.row()[idx:idx+CHUNK_SIZE]
            batch_col_index = self.adj_mat.storage.col()[idx:idx+CHUNK_SIZE]
            a_batch = torch.index_select(a_norm, 0, batch_row_index)
            b_batch = torch.index_select(b_norm, 0, batch_col_index)
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims

    def cal_cos_sim(self,u_idx, i_idx, eps=1e-8, CHUNK_SIZE=1000000):
        user_feature = self.user_emb.weight
        item_feature = self.item_emb.weight
        L = u_idx.shape[0]
        sims = torch.zeros(L, dtype=user_feature.dtype).to(world.device)
        for idx in range(0, L, CHUNK_SIZE):
            a_batch = torch.index_select(user_feature, 0, u_idx[idx:idx + CHUNK_SIZE])
            b_batch = torch.index_select(item_feature, 0, i_idx[idx:idx + CHUNK_SIZE])
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims
 
    def graph_denoising_module(self):
        with torch.no_grad():
            user_emb = self.user_emb.weight
            item_emb = self.item_emb.weight
            all_emb = torch.cat([user_emb,item_emb])
            hidden_emb = matmul(self.edge_index,all_emb)
            hidden_user,hidden_item = torch.split(hidden_emb,[self.num_users,self.num_items])
            # cos_sim = torch.matmul(hidden_user,hidden_item.t())/(torch.norm(hidden_user)*torch.norm(hidden_item))
            # cos_sim = (cos_sim + 1) / 2 # normalize
            # cos_sim[cos_sim < 0.02] = 0
            cos_sim = self.get_cos_sim_sp(hidden_user,hidden_item)
            value = cos_sim
            value =(value + 1)/2 #normalize
            value[value < 0.02] = 0 #pruning
            row_index = torch.cat([self.user_index , self.item_index + self.num_users])
            col_index = torch.cat([self.item_index + self.num_users , self.user_index])
            value = torch.cat([value,value])
            return SparseTensor(row=row_index,col=col_index,value=value,sparse_sizes=(self.num_items + self.num_users , self.num_users + self.num_items))

    """
    Graph Augmentation Module
    input: edge_index(denoised graph)
    output:edge_index(augmented graph)
    """
    def graph_augmentation_module(self):
        with torch.no_grad():
            edge_index = self.graph_denoising_module()
            aug_ratio = 0.1
            pool_multi = 10
            aug_user = torch.LongTensor(np.random.choice(self.num_users,
                                                        int(edge_index.nnz() * aug_ratio * 0.5 * pool_multi))).to(device=world.device)
            aug_item = torch.LongTensor(np.random.choice(self.num_items,
                                                        int(edge_index.nnz() * aug_ratio * 0.5 * pool_multi))).to(world.device)
            cos_sim = self.cal_cos_sim(aug_user,aug_item)
            _, idx = torch.topk(cos_sim, int(edge_index.nnz() * aug_ratio * 0.5))
            aug_user = aug_user[idx].long()
            aug_item = aug_item[idx].long()
            row_index = torch.cat([aug_user , aug_item + self.num_users])
            col_index = torch.cat([aug_item + self.num_users , aug_user])
            aug_value = torch.ones_like(aug_user) * torch.median(edge_index.storage.value())
            aug_value = torch.cat([aug_value,aug_value]).float()
            aug_edge_index = SparseTensor(row=row_index,col=col_index,value=aug_value,sparse_sizes=(self.num_items + self.num_users , self.num_users + self.num_items))
            aug_adj = (edge_index + aug_edge_index).coalesce()
            return aug_adj

    
    def ssl_triple_loss(self, z1: torch.Tensor, z2: torch.Tensor, all_emb: torch.Tensor):
        norm_emb1 = F.normalize(z1)
        norm_emb2 = F.normalize(z2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.mul(norm_emb1, norm_emb2).sum(dim=1)
        ttl_score = torch.matmul(norm_emb1, norm_all_emb.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss
    

    def RGCF_forward(self):
        edge_index = self.graph_denoising_module()
        edge_index_norm = gcn_norm(edge_index)
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
    
    def SSL_forward(self):
        edge_index = self.graph_augmentation_module()
        edge_index_norm = gcn_norm(edge_index)
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
         
    def computer(self):
        users,items = self.RGCF_forward()
        return users,items

    def forward(self,users,items):
        all_users,all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb,items_emb)
        gamma = torch.sum(inner_pro,dim=1)
        return gamma
    
    def getUsersRating_single(self,users):
        all_users,all_items = self.single_computer()
        users_emb = all_users[users.long()].cpu().detach().numpy()
        items_emb = all_items.cpu().detach().numpy()
        rating = self.f(torch.tensor(np.matmul(users_emb,items_emb.T)))
        return rating
    
    def getUsersRating(self,users):
        all_users,all_items = self.computer()
        # users_emb = all_users[users.long()].cpu().detach().numpy()
        # items_emb = all_items.cpu().detach().numpy()
        # #torch.matmul is memory costly
        # rating = self.f(torch.tensor(np.matmul(users_emb,items_emb.T)))
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
    
    def getEmbedding_aug(self,users,pos_items,neg_items):
        all_users,all_items = self.SSL_forward()
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
    
    def ssl_loss(self,users,pos,neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding_aug(users.long(), pos.long(), neg.long())
        aug_user_all_emb,_ = self.SSL_forward()
        aug_u_emb = aug_user_all_emb[users]
        matual_info = self.ssl_triple_loss(users_emb,aug_u_emb,aug_user_all_emb)
        return matual_info * 1e-6  
          
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t,x)

class NSEGCN(MessagePassing):
    def __init__(self,num_users,num_items,embedding_dim = config['latent_dim_rec'] , K = config['K'],add_self_loops = False,edge_index:SparseTensor=None):
        super().__init__()
        self.num_users,self.num_items = num_users,num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.f = nn.Sigmoid()
        self.add_self_loops = add_self_loops
        #E^0 & I^0 -- initial emb
        self.user_emb = nn.Embedding(num_embeddings=self.num_users,embedding_dim=self.embedding_dim)
        self.item_emb = nn.Embedding(num_embeddings=self.num_items,embedding_dim=self.embedding_dim)
        nn.init.normal_(self.user_emb.weight,std=0.1)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        self.edge_index_norm = gcn_norm(edge_index)
        self.edge_index = edge_index
        world.cprint("NSE-GCN will use normal distribution initilizer")

    def computer(self):
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb,items_emb])
        nse_emb = self.propagate(self.edge_index,x=all_emb)
        users_emb,items_emb = torch.split(nse_emb,[self.num_users,self.num_items])
        edge_index_norm = self.edge_index_norm
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
    
    def getUsersRating(self,users):
        all_users,all_items = self.computer()
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


