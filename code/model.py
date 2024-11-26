from torch import nn,Tensor,LongTensor
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import SparseTensor
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import torch
import world
import utils

config = world.config
device = world.device
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

class CenNorm(RecModel):
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
        nn.init.xavier_uniform_(self.user_emb.weight,2)
        nn.init.xavier_uniform_(self.item_emb.weight,2)
        self.K = config['K']
        edge_index = self.get_sparse_graph(edge_index=edge_index,use_value=False,value=None)
        self.edge_index = gcn_norm(edge_index)
        self.alpha= 1./ (1 + self.K)
        self.au = config['au']
        self.r = config['r']
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
            x = self.norm(x)
            out = out + x * self.alpha[i + 1]
        return out
    
    def alignment(self,x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(dim=1).pow(2).mean()

    def alignment_loss(self,edge_label_index):
        out = self.get_embedding()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        batch_x_u,batch_x_i = x_u[edge_label_index[0]],x_i[edge_label_index[1]]
        return self.alignment(batch_x_u,batch_x_i)
    
    def uniformity_loss(self,edge_label_index):
        out = self.get_embedding()
        x_u,x_i = torch.split(out,[self.num_users,self.num_items])
        u_idx = torch.unique(edge_label_index[0])
        i_idx = torch.unique(torch.cat([edge_label_index[1],edge_label_index[2]]))
        batch_x_u,batch_x_i = x_u[u_idx],x_i[i_idx]
        return  self.au * (self.uniformity(batch_x_u) + self.uniformity(batch_x_i))

    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def norm(self,x):
        users,items = torch.split(x,[self.num_users,self.num_items])
        users_norm = torch.clamp(1e-6 + users.pow(2).sum(dim=1).mean(), 1)
        items_norm = torch.clamp(1e-6 + items.pow(2).sum(dim=1).mean(), 1)
        users_norm = users_norm ** self.r
        items_norm = items_norm ** self.r
        users = users / (items_norm)
        items = items / (users_norm)
        x = torch.cat([users,items])
        return x
    
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
        #SGL use normal distribution of 0.01
        nn.init.xavier_normal_(self.user_emb.weight,0.01)
        nn.init.xavier_normal_(self.item_emb.weight,0.01)
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


