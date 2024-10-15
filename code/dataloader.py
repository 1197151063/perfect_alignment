from torch.utils.data import Dataset,DataLoader
from common import cprint
import common
import torch
import numpy as np
from torch_sparse import SparseTensor
import torch_geometric.utils as pyg_utils
from scipy.sparse import csr_matrix
args = common.args

    
class Loader(Dataset):
    """
    Loading data from datasets
    already supportted:['gowalla','amazon-book','yelp2018','lastfm']
    """
    def __init__(self,args=args,path='../data/'):
        dir_path = path + args.dataset
        cprint(f'loading from {dir_path}')
        train_file = dir_path + '/train.txt'
        test_file = dir_path + '/test.txt'
        train_users,train_items = [],[]
        test_users,test_items = [],[]
        train_edge_index,test_edge_index = [],[]
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    all = l.strip('\n').split(' ')
                    uid = int(all[0])
                    val = int(len(all) * 1)
                    items = [int(i) for i in all[1:val]]
                    for item in items:
                        train_edge_index.append([uid,item])
                    train_users.extend([uid] * len(items))
                    train_items.extend(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    all = l.strip('\n').split(' ')
                    uid = int(all[0])
                    try:
                        items = [int(i) for i in all[1:]]
                    except Exception:
                        continue
                    for item in items:
                        test_edge_index.append([uid,item])
                    test_users.extend([uid] * len(items))
                    test_items.extend(items)
        


        train_edge_index = torch.LongTensor(np.array(train_edge_index).T)
        test_edge_index = torch.LongTensor(np.array(test_edge_index).T)
        edge_index = torch.cat((train_edge_index,test_edge_index),1)
        self.edge_index = edge_index
        num_users = len(torch.unique(edge_index[0]))
        num_items = len(torch.unique(edge_index[1]))
        mask = torch.zeros(num_users,num_items)
        
        #self.train_edge_index = train_edge_index
        
        #poisoning the training set
        if args.add_noise == 1:
            nr = args.noise_rate
            row,_ = edge_index
            num_neg = int(row.size(0) * nr)
            neg_indices = pyg_utils.negative_sampling(edge_index,(num_users,num_items),num_neg_samples=num_neg)
            n_r,n_c = neg_indices
            t_r,t_c = train_edge_index
            ind = torch.randint(0,row.size(0),(num_neg,))
            t_r,t_c = t_r.numpy(),t_c.numpy()
            ind = ind.numpy()
            t_r,t_c = t_r.delete(t_r,ind,0),t_c.delete(t_c,ind,0)
            t_r,t_c = torch.from_numpy(t_r),torch.from_numpy(t_c)
            t_r,t_c = torch.cat([t_r,n_r]),torch.cat([t_c,n_c])
            train_edge_index[0],train_edge_index[1] = t_r,t_c
        
        self.num_users = num_users
        self.num_items = num_items
        self.train_edge_index = train_edge_index
        self.UserItemNet = csr_matrix((np.ones(len(self.edge_index[0])), (self.edge_index[0].numpy(), self.edge_index[1].numpy())),
                                      shape=(self.num_users, self.num_items))
        # self.test_edge_index = test_edge_index
        self.bipartite_graph = self.getSparseBipartite()
        self.adj_mat = self.getSparseGraph()

        self.train_loader = DataLoader(
            range(self.train_edge_index.size(1)),
            shuffle=True,
            batch_size=args.bpr_batch
        )
        self.test_loader = DataLoader(
            list(range(num_users)),batch_size=args.testbatch,shuffle=False,num_workers=5
        )
        test_ground_truth_list = [[] for _ in range(num_users)]
        for i in range(len(test_items)):
            test_ground_truth_list[test_users[i]].append(test_items[i])
        for i in range(len(train_items)):
            mask[train_users[i]][train_items[i]] = -np.inf
        self.test_ground_truth_list = test_ground_truth_list
        self.mask = mask
    '''
    A = |0   R|
        |R^T 0|
    R : user-item bipartite graph
    A : unnormalized Adjacency Matrix
    '''
    def getSparseGraph(self):
        cprint("generate Adjacency Matrix A")
        user_index = self.train_edge_index[0]
        item_index = self.train_edge_index[1]
        row_index = torch.cat([user_index,item_index+self.num_users])
        col_index = torch.cat([item_index+self.num_users,user_index])
        return SparseTensor(row=row_index,col=col_index,sparse_sizes=(self.num_items+self.num_users,self.num_items+self.num_users))

    def getSparseBipartite(self):
        user_index = self.train_edge_index[0]
        item_index = self.train_edge_index[1]
        return SparseTensor(row=user_index,col=item_index,sparse_sizes=(self.num_users,self.num_items))
    
    def get_user_all_interacted(self,users):
        users = users.detach().cpu().numpy()
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
