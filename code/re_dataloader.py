from torch.utils.data import Dataset,DataLoader
from world import cprint
import world
import torch
import numpy as np
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
seed = world.seed
import random
import torch_geometric.utils as pyg_utils

class BasicDataset(Dataset):
    def __init__(self):
        cprint('init dataset')
    
    @property
    def num_users(self):
        raise NotImplementedError
    
    @property
    def num_items(self):
        raise NotImplementedError
    
    def getSparseGraph(self):
        raise NotImplementedError
    
class Loader(BasicDataset):
    """
    Loading data from datasets
    supporting:['gowalla','amazon-book','yelp2018','lastfm']
    """
    def __init__(self,config=world.config,path='../data/'):
        dir_path = path + config['dataset']
        cprint(f'loading from {dir_path}')
        self.n_user = 0
        self.n_item = 0
        train_file = dir_path + '/train.txt'
        test_file = dir_path + '/test.txt'
        train_edge_index = []
        test_edge_index = []
        val_edge_index = []
        testUser = []
        testItem = []
        trainUser = []
        trainItem = []
        valUser = []
        valItem = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    all = l.strip('\n').split(' ')
                    val = int(1 * len(all))
                    uid = int(all[0])
                    train_items = [int(i) for i in all[1:]]
                    val_items = [int(i) for i in all[1:]]
                    for item in train_items:
                        train_edge_index.append([uid,item])
                    for item in val_items:
                        val_edge_index.append([uid,item])
                    trainUser.extend([uid] * len(train_items))
                    trainItem.extend(train_items)
                    valUser.extend([uid] * len(val_items))
                    valItem.extend(val_items)
        train_edge_index = torch.LongTensor(np.array(train_edge_index).T)
        val_edge_index = torch.LongTensor(np.array(val_edge_index).T)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    all = l.strip('\n').split(' ')
                    uid = int(all[0])
                    try:
                        items = [int(i) for i in all[1:]]
                        for item in items:
                            test_edge_index.append([uid,item])
                    except Exception:
                        continue
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
        test_edge_index = torch.LongTensor(np.array(test_edge_index).T)


        edge_index = torch.cat((train_edge_index,test_edge_index),1)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.valUser = np.array(valUser)
        self.valItem = np.array(valItem)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.train_edge_index = train_edge_index
        self.edge_index = edge_index
        self.n_user = len(np.unique(edge_index[0]))
        self.n_item = len(np.unique(edge_index[1]))
        self.UserItemNet = csr_matrix((np.ones(len(self.train_edge_index[0])), (self.train_edge_index[0].numpy(), self.train_edge_index[1].numpy())),
                                      shape=(self.n_user, self.n_item))
        UserItemValid = csr_matrix((np.ones(len(val_edge_index[0])), (val_edge_index[0].numpy(), val_edge_index[1].numpy())),
                                      shape=(self.n_user, self.n_item))
        #poisoning the training and validation set
        if config['add_noise'] == 1:
            graph = self.UserItemNet.toarray()
            graph_val = UserItemValid.toarray()
            nr = config['noise_rate']
            cprint(f'adding noise at noise rate {nr}')
            for user in range(self.n_user):
                maxlen = int(nr * graph[user][graph[user]==1].size) + 1
                zero_ind = np.where(graph[user]==0)
                one_ind = np.where(graph[user]==1)
                sample_zero = np.random.choice(zero_ind[0],maxlen,replace=False)
                sample_one = np.random.choice(one_ind[0],maxlen,replace=False)
                graph[user][sample_zero] = 1
                graph[user][sample_one] = 0

            for user in range(self.n_user):
                maxlen = int(nr * graph_val[user][graph_val[user]==1].size) + 1
                zero_ind = np.where(graph_val[user]==0)
                one_ind = np.where(graph_val[user]==1)
                if len(one_ind) == 0:
                    continue
                else:
                    sample_zero = np.random.choice(zero_ind[0],maxlen,replace=False)
                    sample_one = np.random.choice(one_ind[0],maxlen,replace=False)
                    graph_val[user][sample_zero] = 1
                    graph_val[user][sample_one] = 0


            self.UserItemNet = csr_matrix(graph)
            row_indices, col_indices = self.UserItemNet.nonzero()
            self.train_edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long,device=world.device)
            UserItemValid = csr_matrix(graph_val).tocoo()
            self.valUser = UserItemValid.row
            self.valItem = UserItemValid.col
        '''
        the User-Item bipartite graph is modified
        rebuild the train_edge_index
        '''
        """
        build all positive dict to boost negative sampling
        """

        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.test_edge_index = test_edge_index

        # self.train_loader=DataLoader(self.train_edge_index.size(1),
        #                              batch_size=4096,
        #                              shuffle=True)
        # self.test_loader=DataLoader(list(range(self.num_users)),
        #                             batch_size=config['testbatch'],
        #                             shuffle=False,
        #                             num_workers=4)
        self.__testDict = self.__build_test()
        self.__valDict = self.__build_val()
        print(f"{world.dataset} is ready to go")

    @property
    def num_users(self):
        return self.n_user
    @property
    def num_items(self):
        return self.n_item
    @property
    def n_users(self):
        return self.n_user
    @property
    def m_items(self):
        return self.n_item
    @property
    def testDict(self):
        return self.__testDict
    @property
    def valDict(self):
        return self.__valDict
    @property
    def allPos(self):
        return self._allPos
    @property
    def trainDataSize(self):
        return len(self.train_edge_index[1])
    

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
        row_index = torch.cat([user_index,item_index+self.n_user])
        col_index = torch.cat([item_index+self.n_user,user_index])
        return SparseTensor(row=row_index,col=col_index,sparse_sizes=(self.n_item+self.n_user,self.n_item+self.n_user))

    def getSparseBipartite(self):
        user_index = self.train_edge_index[0]
        item_index = self.train_edge_index[1]
        return SparseTensor(row=user_index,col=item_index,sparse_sizes=(self.num_users,self.num_items))
    
    def getUserPosItems(self,users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_val(self):
        """ validation
        return:
            dict: {user: [items]}
        """
        val_data = {}
        for i, item in enumerate(self.valItem):
            user = self.valUser[i]
            if val_data.get(user):
                val_data[user].append(item)
            else:
                val_data[user] = [item]
        return val_data
