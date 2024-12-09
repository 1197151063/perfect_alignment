import world
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from torch import LongTensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from random import shuffle,choice
device = world.device
config = world.config


def next_batch_pairwise(dataset):
    batch_size = config['bpr_batch_size']
    train_edge_index = dataset.train_edge_index
    index = torch.arange(train_edge_index.size(1))
    allPos = dataset.allPos
    # training_data = data.training_data
    shuffle(index)
    ptr = 0
    data_size = train_edge_index.size(1)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [train_edge_index[0][idx] for idx in range(ptr, batch_end)]
        items = [train_edge_index[1][idx] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        pos_for_users = allPos[users]
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(1):
                neg_item = choice(item_list)
                while neg_item in pos_for_users[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx

def Fast_Sampling(dataset):
    """
    A more efficient sampler with simplified negative sampling
    easy to overfit on raw GNN model
    """
    train_edge_index = dataset.train_edge_index.to(device)
    num_items = dataset.num_items
    batch_size = config['bpr_batch_size']
    mini_batch = []
    train_loader = DataLoader(
            range(train_edge_index.size(1)),
            shuffle=True,
            batch_size=batch_size)
    for index in train_loader:
        pos_edge_label_index = train_edge_index[:,index]
        neg_edge_label_index = torch.randint(0, num_items,(index.numel(), ), device=device)
        edge_label_index = torch.stack([
            pos_edge_label_index[0],
            pos_edge_label_index[1],
            neg_edge_label_index,
        ])
        mini_batch.append(edge_label_index)
    return mini_batch
        
    
    





def Sampling(dataset,all_pos):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    S = []
    for user in users:
        posForUser = all_pos[user]
        if len(posForUser) == 0:
            continue
        positem = np.random.choice(posForUser)
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem not in posForUser:
                break
        S.append([user, positem, negitem])
    return np.array(S)


# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def edge_drop(edge_index:LongTensor,drop_ratio = 0.1):
        num_drop = int(drop_ratio * edge_index.size(1))
        drop_index = np.random.randint(0,edge_index.size(1),(num_drop,))
        drop_index = torch.tensor(drop_index).to(device)
        mask = torch.ones_like(edge_index[0],dtype=torch.bool,device=edge_index.device)
        mask[drop_index[:num_drop]] = False
        edge_index_new = edge_index[:,mask]
        return edge_index_new

def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

def early_stopping(recall,
                   ndcg,
                   best,
                   patience,
                   model):
    if patience < 50: 
        if recall + ndcg > best: 
            patience = 0
            print('[BEST]')
            best = recall + ndcg
            # torch.save(model.state_dict(), save_file_name)
            # torch.save(model.state_dict(),'./models/' + save_file_name)
        else:
            patience += 1
        return 0,best,patience
    else:
        return 1,best,patience # Perform Early Stopping 
# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def eval(node_count,topk_index,logits,ground_truth,k):
    isin_mat = ground_truth.gather(1, topk_index)
    # Calculate recall
    recall = float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
    # Calculate NDCG
    log_positions = torch.log2(torch.arange(2, k + 2, device=logits.device).float())
    dcg = (isin_mat / log_positions).sum(dim=-1)
    ideal_dcg = torch.zeros_like(dcg)
    for i in range(len(dcg)):
        ideal_dcg = (1.0 / log_positions[:node_count[i].clamp(max=k).int()]).sum()
    ndcg = float((dcg / ideal_dcg.clamp(min=1e-6)).sum())
    return recall,ndcg

# ====================end Metrics=============================
# =========================================================
